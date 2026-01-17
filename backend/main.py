from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session
from typing import Optional, AsyncGenerator, Literal
from collections import defaultdict
from enum import Enum
import anthropic
import openai
import google.generativeai as genai
import os
import io
import re
import time
import html
import json
import secrets
from pathlib import Path
from dotenv import load_dotenv

# File parsing
from pypdf import PdfReader
from docx import Document

from database import get_db, init_db, Course, AdminUser, LLMSettings, WidgetStyle, SessionLocal, DEFAULT_WIDGET_CONFIG

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

# Security settings from environment
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "20"))
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", "500000"))
MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", "5000"))
MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", "20"))

# Admin authentication - credentials stored in database
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# HTTP Basic Auth setup
security = HTTPBasic()


def is_setup_complete() -> bool:
    """Check if admin account has been created."""
    db = SessionLocal()
    try:
        admin = db.query(AdminUser).first()
        return admin is not None
    finally:
        db.close()


def verify_admin_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """
    Verify admin credentials against the database.
    Returns the username if valid, raises 401 if not.
    """
    db = SessionLocal()
    try:
        admin = db.query(AdminUser).filter(AdminUser.username == credentials.username).first()

        if not admin or not admin.verify_password(credentials.password):
            raise HTTPException(
                status_code=401,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Basic realm=\"Admin Access\""}
            )

        return credentials.username
    finally:
        db.close()

# =============================================================================
# LLM PROVIDER CONFIGURATION
# =============================================================================

# Supported LLM providers and their models
LLM_PROVIDERS = {
    "anthropic": {
        "name": "Anthropic (Claude)",
        "env_key": "ANTHROPIC_API_KEY",
        "models": [
            {"id": "claude-opus-4-5-20251101", "name": "Claude Opus 4.5", "description": "Most intelligent, best for coding and agents"},
            {"id": "claude-sonnet-4-5-20250929", "name": "Claude Sonnet 4.5", "description": "Best balance of speed and intelligence"},
            {"id": "claude-haiku-4-5-20251001", "name": "Claude Haiku 4.5", "description": "Fastest, most cost-effective"},
        ]
    },
    "openai": {
        "name": "OpenAI (GPT)",
        "env_key": "OPENAI_API_KEY",
        "models": [
            {"id": "gpt-5.2", "name": "GPT-5.2", "description": "Most capable, best for professional work"},
            {"id": "gpt-5.2-pro", "name": "GPT-5.2 Pro", "description": "Extended thinking for complex tasks"},
            {"id": "gpt-5.2-chat-latest", "name": "GPT-5.2 Instant", "description": "Fast and cost-effective"},
        ]
    },
    "google": {
        "name": "Google (Gemini)",
        "env_key": "GOOGLE_API_KEY",
        "models": [
            {"id": "gemini-3-pro", "name": "Gemini 3 Pro", "description": "Most capable, advanced reasoning"},
            {"id": "gemini-3-flash", "name": "Gemini 3 Flash", "description": "Pro-level intelligence at Flash speed"},
            {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash", "description": "Previous generation, stable"},
        ]
    }
}

def get_available_providers():
    """Return list of providers that have API keys configured."""
    available = []
    for provider_id, config in LLM_PROVIDERS.items():
        api_key = os.getenv(config["env_key"])
        if api_key:
            available.append({
                "id": provider_id,
                "name": config["name"],
                "models": config["models"]
            })
    return available

def get_configured_provider_and_model():
    """
    Get the globally configured LLM provider and model from environment variables.
    Falls back to first available provider if not explicitly configured.
    """
    # Check for explicit configuration
    configured_provider = os.getenv("LLM_PROVIDER")
    configured_model = os.getenv("LLM_MODEL")

    available = get_available_providers()

    if configured_provider and configured_model:
        # Validate the configured provider has an API key
        if any(p["id"] == configured_provider for p in available):
            return configured_provider, configured_model

    # Fall back to first available provider
    if available:
        return available[0]["id"], available[0]["models"][0]["id"]

    return "anthropic", "claude-sonnet-4-5-20250929"  # Ultimate fallback


def get_api_key_for_provider(provider: str) -> Optional[str]:
    """Get the API key for a specific provider from database or environment."""
    db = SessionLocal()
    try:
        settings = db.query(LLMSettings).filter(LLMSettings.id == "global").first()

        if provider == "anthropic":
            return (settings and settings.anthropic_api_key) or os.getenv("ANTHROPIC_API_KEY")
        elif provider == "openai":
            return (settings and settings.openai_api_key) or os.getenv("OPENAI_API_KEY")
        elif provider == "google":
            return (settings and settings.google_api_key) or os.getenv("GOOGLE_API_KEY")
        return None
    finally:
        db.close()


# =============================================================================
# LLM CLIENT FUNCTIONS
# =============================================================================

def call_anthropic(messages: list, system_prompt: str, model: str, stream: bool = False):
    """Call Anthropic Claude API."""
    api_key = get_api_key_for_provider("anthropic")
    if not api_key:
        raise HTTPException(status_code=500, detail="Anthropic API key not configured")

    client = anthropic.Anthropic(api_key=api_key)

    if stream:
        return client.messages.stream(
            model=model,
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            messages=messages
        )
    else:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            messages=messages
        )
        return response.content[0].text


def call_openai(messages: list, system_prompt: str, model: str, stream: bool = False):
    """Call OpenAI GPT API."""
    api_key = get_api_key_for_provider("openai")
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

    client = openai.OpenAI(api_key=api_key)

    # Convert messages format and prepend system message
    openai_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        openai_messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    if stream:
        return client.chat.completions.create(
            model=model,
            messages=openai_messages,
            max_tokens=1024,
            stream=True
        )
    else:
        response = client.chat.completions.create(
            model=model,
            messages=openai_messages,
            max_tokens=1024
        )
        return response.choices[0].message.content


def call_google(messages: list, system_prompt: str, model: str, stream: bool = False):
    """Call Google Gemini API."""
    api_key = get_api_key_for_provider("google")
    if not api_key:
        raise HTTPException(status_code=500, detail="Google API key not configured")

    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(model, system_instruction=system_prompt)

    # Convert messages to Gemini format (history + current message)
    history = []
    current_message = ""

    for i, msg in enumerate(messages):
        role = "user" if msg["role"] == "user" else "model"
        if i == len(messages) - 1:
            current_message = msg["content"]
        else:
            history.append({"role": role, "parts": [msg["content"]]})

    chat = gemini_model.start_chat(history=history)

    if stream:
        return chat.send_message(current_message, stream=True)
    else:
        response = chat.send_message(current_message)
        return response.text


def get_llm_response(provider: str, model: str, messages: list, system_prompt: str, stream: bool = False):
    """Unified LLM call interface."""
    if provider == "anthropic":
        return call_anthropic(messages, system_prompt, model, stream)
    elif provider == "openai":
        return call_openai(messages, system_prompt, model, stream)
    elif provider == "google":
        return call_google(messages, system_prompt, model, stream)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported LLM provider: {provider}")


app = FastAPI(
    title="Course Chat Widget API",
    version="1.0.0",
    docs_url="/docs" if DEBUG else None,
    redoc_url="/redoc" if DEBUG else None,
)

# =============================================================================
# RATE LIMITING
# =============================================================================

class RateLimiter:
    """Simple in-memory rate limiter by IP address."""

    def __init__(self, requests_per_minute: int = 20):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)

    def is_allowed(self, ip: str) -> bool:
        """Check if the IP is allowed to make a request."""
        now = time.time()
        minute_ago = now - 60

        # Clean old requests
        self.requests[ip] = [t for t in self.requests[ip] if t > minute_ago]

        # Check rate limit
        if len(self.requests[ip]) >= self.requests_per_minute:
            return False

        # Record this request
        self.requests[ip].append(now)
        return True

    def get_remaining(self, ip: str) -> int:
        """Get remaining requests for this IP."""
        now = time.time()
        minute_ago = now - 60
        self.requests[ip] = [t for t in self.requests[ip] if t > minute_ago]
        return max(0, self.requests_per_minute - len(self.requests[ip]))

rate_limiter = RateLimiter(requests_per_minute=RATE_LIMIT_PER_MINUTE)

# =============================================================================
# SECURITY HELPERS
# =============================================================================

def sanitize_string(s: str, max_length: int = 10000) -> str:
    """Sanitize a string by escaping HTML and limiting length."""
    if not s:
        return s
    # Truncate to max length
    s = s[:max_length]
    # Remove null bytes and other control characters (except newlines/tabs)
    s = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', s)
    return s

def validate_uuid(value: str) -> bool:
    """Validate that a string looks like a UUID."""
    uuid_pattern = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        re.IGNORECASE
    )
    return bool(uuid_pattern.match(value))

def get_client_ip(request: Request) -> str:
    """Get the client IP, handling proxies."""
    # Check for forwarded IP (when behind a proxy)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # Take the first IP in the chain (original client)
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent


def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from a PDF file."""
    pdf = PdfReader(io.BytesIO(file_content))
    text = []
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return "\n\n".join(text)


def extract_text_from_docx(file_content: bytes) -> str:
    """Extract text from a DOCX file."""
    doc = Document(io.BytesIO(file_content))
    text = []
    for para in doc.paragraphs:
        if para.text.strip():
            text.append(para.text)
    return "\n\n".join(text)


def extract_text_from_file(filename: str, content: bytes) -> str:
    """Extract text based on file type."""
    filename_lower = filename.lower()
    
    if filename_lower.endswith('.pdf'):
        return extract_text_from_pdf(content)
    elif filename_lower.endswith('.docx'):
        return extract_text_from_docx(content)
    elif filename_lower.endswith('.txt') or filename_lower.endswith('.md'):
        return content.decode('utf-8')
    else:
        raise ValueError(f"Unsupported file type: {filename}")

# =============================================================================
# MIDDLEWARE
# =============================================================================

# CORS - configurable via ALLOWED_ORIGINS environment variable
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)

    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

    # Allow widget.html to be embedded in iframes (for Skilljar integration)
    # All other pages are restricted to same-origin only
    if request.url.path == "/widget.html":
        # Remove X-Frame-Options to allow embedding anywhere
        # The widget is designed to be embedded in external LMS pages
        pass
    else:
        response.headers["X-Frame-Options"] = "SAMEORIGIN"

    # Only add HSTS in production (when not using localhost)
    if not request.url.hostname or "localhost" not in request.url.hostname:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

    return response

# Initialize database on startup
@app.on_event("startup")
def startup():
    init_db()


# Pydantic models
class WidgetColors(BaseModel):
    primary: str = "#0D2B3E"
    accent: str = "#5FCFB4"
    background: str = "#F8FAFC"
    messageBg: str = "#FFFFFF"
    messageText: str = "#334155"


class WidgetText(BaseModel):
    title: str = "Course Assistant"
    subtitle: str = "Ask questions about this course"
    placeholder: str = "Type your question..."
    welcomeMessage: str = "Welcome! I'm here to help answer any questions you have about this course. What would you like to know?"


class WidgetStyleSettings(BaseModel):
    borderRadius: str = "8"
    iconStyle: str = "minimal"  # minimal, rounded, square
    widgetTheme: str = "light"  # light, dark
    position: str = "bottom-right"  # bottom-right, bottom-left


class WidgetConfig(BaseModel):
    colors: WidgetColors = WidgetColors()
    text: WidgetText = WidgetText()
    style: WidgetStyleSettings = WidgetStyleSettings()


class CourseCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    content: str = Field(..., min_length=1)
    system_prompt: Optional[str] = Field(None, max_length=5000)
    widget_config: Optional[WidgetConfig] = None
    lms_course_id: Optional[str] = Field(None, max_length=255)  # External LMS course ID
    llm_provider: Optional[str] = Field(None, max_length=50)  # anthropic, openai, google
    llm_model: Optional[str] = Field(None, max_length=100)  # specific model ID

    @validator('name')
    def sanitize_name(cls, v):
        return sanitize_string(v, max_length=200)

    @validator('description')
    def sanitize_description(cls, v):
        if v:
            return sanitize_string(v, max_length=1000)
        return v

    @validator('content')
    def validate_content(cls, v):
        if len(v) > MAX_CONTENT_LENGTH:
            raise ValueError(f'Content exceeds maximum length of {MAX_CONTENT_LENGTH} characters')
        return sanitize_string(v, max_length=MAX_CONTENT_LENGTH)

    @validator('lms_course_id')
    def sanitize_lms_course_id(cls, v):
        if v:
            return sanitize_string(v, max_length=255).strip()
        return v

    @validator('llm_provider')
    def validate_llm_provider(cls, v):
        if v and v not in LLM_PROVIDERS:
            raise ValueError(f'Invalid LLM provider. Must be one of: {", ".join(LLM_PROVIDERS.keys())}')
        return v


class CourseUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    content: Optional[str] = None
    system_prompt: Optional[str] = Field(None, max_length=5000)
    widget_config: Optional[WidgetConfig] = None
    lms_course_id: Optional[str] = Field(None, max_length=255)  # External LMS course ID
    llm_provider: Optional[str] = Field(None, max_length=50)  # anthropic, openai, google
    llm_model: Optional[str] = Field(None, max_length=100)  # specific model ID

    @validator('content')
    def validate_content(cls, v):
        if v and len(v) > MAX_CONTENT_LENGTH:
            raise ValueError(f'Content exceeds maximum length of {MAX_CONTENT_LENGTH} characters')
        if v:
            return sanitize_string(v, max_length=MAX_CONTENT_LENGTH)
        return v

    @validator('lms_course_id')
    def sanitize_lms_course_id(cls, v):
        if v:
            return sanitize_string(v, max_length=255).strip()
        return v

    @validator('llm_provider')
    def validate_llm_provider(cls, v):
        if v and v not in LLM_PROVIDERS:
            raise ValueError(f'Invalid LLM provider. Must be one of: {", ".join(LLM_PROVIDERS.keys())}')
        return v


class CourseResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    content: str
    system_prompt: Optional[str]
    widget_config: Optional[dict]
    lms_course_id: Optional[str]
    llm_provider: Optional[str]
    llm_model: Optional[str]
    embed_code: str

    class Config:
        from_attributes = True


class ChatRequest(BaseModel):
    course_id: str = Field(..., min_length=36, max_length=36)
    message: str = Field(..., min_length=1)
    conversation_history: Optional[list] = []

    @validator('course_id')
    def validate_course_id(cls, v):
        if not validate_uuid(v):
            raise ValueError('Invalid course ID format')
        return v

    @validator('message')
    def validate_message(cls, v):
        if len(v) > MAX_MESSAGE_LENGTH:
            raise ValueError(f'Message exceeds maximum length of {MAX_MESSAGE_LENGTH} characters')
        return sanitize_string(v, max_length=MAX_MESSAGE_LENGTH)

    @validator('conversation_history')
    def validate_history(cls, v):
        if v and len(v) > MAX_CONVERSATION_HISTORY:
            # Truncate to most recent messages
            return v[-MAX_CONVERSATION_HISTORY:]
        return v


class ChatResponse(BaseModel):
    response: str


# =============================================================================
# LLM PROVIDERS ENDPOINT
# =============================================================================

@app.get("/api/providers")
def get_providers():
    """Get the globally configured LLM provider and model."""
    provider, model = get_configured_provider_and_model()
    available = get_available_providers()

    # Find the provider details
    provider_info = next((p for p in available if p["id"] == provider), None)
    model_info = None
    if provider_info:
        model_info = next((m for m in provider_info["models"] if m["id"] == model), None)

    return {
        "provider": provider,
        "provider_name": provider_info["name"] if provider_info else provider,
        "model": model,
        "model_name": model_info["name"] if model_info else model,
        "model_description": model_info["description"] if model_info else ""
    }


# Course CRUD endpoints
@app.post("/api/courses", response_model=CourseResponse)
def create_course(course: CourseCreate, db: Session = Depends(get_db), username: str = Depends(verify_admin_credentials)):
    db_course = Course(
        name=course.name,
        description=course.description,
        content=course.content,
        system_prompt=course.system_prompt,
        widget_config=course.widget_config.dict() if course.widget_config else None,
        lms_course_id=course.lms_course_id,
        llm_provider=course.llm_provider,
        llm_model=course.llm_model
    )
    db.add(db_course)
    db.commit()
    db.refresh(db_course)

    return _course_to_response(db_course)


@app.get("/api/courses")
def list_courses(db: Session = Depends(get_db)):
    courses = db.query(Course).all()
    return [_course_to_response(c) for c in courses]


@app.get("/api/courses/{course_id}", response_model=CourseResponse)
def get_course(course_id: str, db: Session = Depends(get_db)):
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    return _course_to_response(course)


@app.put("/api/courses/{course_id}", response_model=CourseResponse)
def update_course(course_id: str, course_update: CourseUpdate, db: Session = Depends(get_db), username: str = Depends(verify_admin_credentials)):
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    if course_update.name is not None:
        course.name = course_update.name
    if course_update.description is not None:
        course.description = course_update.description
    if course_update.content is not None:
        course.content = course_update.content
    if course_update.system_prompt is not None:
        course.system_prompt = course_update.system_prompt
    if course_update.widget_config is not None:
        course.widget_config = course_update.widget_config.dict()
    if course_update.lms_course_id is not None:
        course.lms_course_id = course_update.lms_course_id
    if course_update.llm_provider is not None:
        course.llm_provider = course_update.llm_provider
    if course_update.llm_model is not None:
        course.llm_model = course_update.llm_model

    db.commit()
    db.refresh(course)
    return _course_to_response(course)


@app.delete("/api/courses/{course_id}")
def delete_course(course_id: str, db: Session = Depends(get_db), username: str = Depends(verify_admin_credentials)):
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    
    db.delete(course)
    db.commit()
    return {"message": "Course deleted successfully"}


# File upload endpoint
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), username: str = Depends(verify_admin_credentials)):
    """Upload a file and extract its text content."""
    try:
        content = await file.read()
        
        if len(content) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")
        
        text = extract_text_from_file(file.filename, content)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from file.")
        
        return {
            "filename": file.filename,
            "text": text,
            "character_count": len(text),
            "word_count": len(text.split())
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


# Chat endpoint
@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest, req: Request, db: Session = Depends(get_db)):
    # Rate limiting
    client_ip = get_client_ip(req)
    if not rate_limiter.is_allowed(client_ip):
        remaining = rate_limiter.get_remaining(client_ip)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Please wait before sending more messages.",
            headers={"Retry-After": "60", "X-RateLimit-Remaining": str(remaining)}
        )

    course = db.query(Course).filter(Course.id == request.course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    # Get LLM settings: use course-specific if set, otherwise fall back to global
    if course.llm_provider and course.llm_model:
        provider = course.llm_provider
        model = course.llm_model
    else:
        provider, model = get_configured_provider_and_model()

    # Build the system prompt - Smart prompt that stays on-topic but allows helpful context
    base_system = course.system_prompt or f"""You are a helpful learning assistant for the course "{course.name}".

PRIMARY ROLE: Help learners understand and succeed in this course.

GUIDELINES:
1. COURSE CONTENT IS YOUR PRIMARY SOURCE - Always ground your answers in the course material provided below when relevant.

2. SUPPLEMENT WITH CONTEXT - You may use your general knowledge to:
   - Clarify or explain course concepts in different ways
   - Provide helpful examples that reinforce course material
   - Answer follow-up questions that deepen understanding of course topics
   - Connect course concepts to real-world applications

3. STAY ON TOPIC - If a learner asks something completely unrelated to the course subject matter (like recipes, sports scores, personal advice, etc.), respond warmly but redirect:
   "I'm here to help you with {course.name}! Is there something about the course I can help you understand?"

4. BE A GREAT LEARNING AID - Your goal is to help learners truly understand the material, not just recite it. Explain concepts clearly, encourage questions, and help them connect ideas.

5. WHEN UNCERTAIN - If the course content doesn't cover something but it's related to the topic, you can share your knowledge while noting: "This isn't specifically covered in the course, but here's what I can share..."

COURSE NAME: {course.name}"""

    system_prompt = f"""{base_system}

=== COURSE CONTENT ===
{course.content}
=== END COURSE CONTENT ==="""

    # Build messages with conversation history
    messages = []
    for msg in request.conversation_history:
        messages.append({
            "role": msg.get("role", "user"),
            "content": msg.get("content", "")
        })

    messages.append({
        "role": "user",
        "content": request.message
    })

    try:
        # Use the unified LLM interface
        response_text = get_llm_response(provider, model, messages, system_prompt, stream=False)
        return ChatResponse(response=response_text)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with AI: {str(e)}")


# =============================================================================
# STREAMING CHAT ENDPOINT
# =============================================================================

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest, req: Request, db: Session = Depends(get_db)):
    """Stream chat responses for better UX."""
    # Rate limiting
    client_ip = get_client_ip(req)
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please wait before sending more messages.",
            headers={"Retry-After": "60"}
        )

    course = db.query(Course).filter(Course.id == request.course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    # Get LLM settings: use course-specific if set, otherwise fall back to global
    if course.llm_provider and course.llm_model:
        provider = course.llm_provider
        model = course.llm_model
    else:
        provider, model = get_configured_provider_and_model()

    # Build the system prompt
    base_system = course.system_prompt or f"""You are a helpful learning assistant for the course "{course.name}".

PRIMARY ROLE: Help learners understand and succeed in this course.

GUIDELINES:
1. COURSE CONTENT IS YOUR PRIMARY SOURCE - Always ground your answers in the course material provided below when relevant.

2. SUPPLEMENT WITH CONTEXT - You may use your general knowledge to:
   - Clarify or explain course concepts in different ways
   - Provide helpful examples that reinforce course material
   - Answer follow-up questions that deepen understanding of course topics
   - Connect course concepts to real-world applications

3. STAY ON TOPIC - If a learner asks something completely unrelated to the course subject matter (like recipes, sports scores, personal advice, etc.), respond warmly but redirect:
   "I'm here to help you with {course.name}! Is there something about the course I can help you understand?"

4. BE A GREAT LEARNING AID - Your goal is to help learners truly understand the material, not just recite it. Explain concepts clearly, encourage questions, and help them connect ideas.

5. WHEN UNCERTAIN - If the course content doesn't cover something but it's related to the topic, you can share your knowledge while noting: "This isn't specifically covered in the course, but here's what I can share..."

COURSE NAME: {course.name}"""

    system_prompt = f"""{base_system}

=== COURSE CONTENT ===
{course.content}
=== END COURSE CONTENT ==="""

    # Build messages with conversation history
    messages = []
    for msg in request.conversation_history:
        messages.append({
            "role": msg.get("role", "user"),
            "content": msg.get("content", "")
        })

    messages.append({
        "role": "user",
        "content": request.message
    })

    async def generate_anthropic():
        """Stream from Anthropic Claude."""
        api_key = get_api_key_for_provider("anthropic")
        if not api_key:
            yield f"data: {json.dumps({'error': 'Anthropic API key not configured'})}\n\n"
            return

        client = anthropic.Anthropic(api_key=api_key)
        try:
            with client.messages.stream(
                model=model,
                max_tokens=1024,
                system=[
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"}
                    }
                ],
                messages=messages
            ) as stream:
                for text in stream.text_stream:
                    yield f"data: {json.dumps({'text': text})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    async def generate_openai():
        """Stream from OpenAI GPT."""
        api_key = get_api_key_for_provider("openai")
        if not api_key:
            yield f"data: {json.dumps({'error': 'OpenAI API key not configured'})}\n\n"
            return

        client = openai.OpenAI(api_key=api_key)
        openai_messages = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            openai_messages.append({"role": msg["role"], "content": msg["content"]})

        try:
            stream = client.chat.completions.create(
                model=model,
                messages=openai_messages,
                max_tokens=1024,
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield f"data: {json.dumps({'text': chunk.choices[0].delta.content})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    async def generate_google():
        """Stream from Google Gemini."""
        api_key = get_api_key_for_provider("google")
        if not api_key:
            yield f"data: {json.dumps({'error': 'Google API key not configured'})}\n\n"
            return

        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel(model, system_instruction=system_prompt)

        # Convert messages to Gemini format
        history = []
        current_message = ""
        for i, msg in enumerate(messages):
            role = "user" if msg["role"] == "user" else "model"
            if i == len(messages) - 1:
                current_message = msg["content"]
            else:
                history.append({"role": role, "parts": [msg["content"]]})

        chat = gemini_model.start_chat(history=history)

        try:
            response = chat.send_message(current_message, stream=True)
            for chunk in response:
                if chunk.text:
                    yield f"data: {json.dumps({'text': chunk.text})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    # Select the appropriate generator based on provider
    if provider == "anthropic":
        generator = generate_anthropic()
    elif provider == "openai":
        generator = generate_openai()
    elif provider == "google":
        generator = generate_google()
    else:
        async def error_generator():
            yield f"data: {json.dumps({'error': f'Unsupported provider: {provider}'})}\n\n"
        generator = error_generator()

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# =============================================================================
# SUGGESTED QUESTIONS ENDPOINT
# =============================================================================

@app.get("/api/courses/{course_id}/suggestions")
def get_suggested_questions(course_id: str, db: Session = Depends(get_db)):
    """Generate suggested questions based on course content."""
    if not validate_uuid(course_id):
        raise HTTPException(status_code=400, detail="Invalid course ID format")

    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    # Get global LLM settings from environment
    provider, model = get_configured_provider_and_model()

    # Extract a preview of the content for generating suggestions
    content_preview = course.content[:3000] if len(course.content) > 3000 else course.content

    system_prompt = "You are a helpful assistant. Generate exactly 3 short, natural questions that a learner might ask about the course content. Questions should be simple and inviting - not too complex. Return ONLY a JSON array of 3 strings, no other text."

    user_message = f"""Based on this course content, generate 3 starter questions a learner might ask. Keep them short (under 50 characters each) and conversational.

Course: {course.name}

Content preview:
{content_preview}

Return only a JSON array like: ["Question 1?", "Question 2?", "Question 3?"]"""

    messages = [{"role": "user", "content": user_message}]

    try:
        response_text = get_llm_response(provider, model, messages, system_prompt, stream=False)

        # Handle potential markdown code blocks
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()

        suggestions = json.loads(response_text)

        # Validate it's a list of strings
        if not isinstance(suggestions, list):
            raise ValueError("Response is not a list")

        suggestions = [s for s in suggestions if isinstance(s, str)][:3]

        return {"suggestions": suggestions}

    except json.JSONDecodeError:
        # Fallback: generate generic questions based on course name
        return {
            "suggestions": [
                f"What is {course.name} about?",
                "What are the main topics covered?",
                "Where should I start?"
            ]
        }
    except Exception as e:
        # Fallback on any error
        return {
            "suggestions": [
                f"What is {course.name} about?",
                "What will I learn in this course?",
                "Can you summarize the key points?"
            ]
        }


# =============================================================================
# WIDGET DETECTION ENDPOINT (for floating widget)
# =============================================================================
# NOTE: This route MUST come before /api/widget/{course_id} to avoid path parameter matching

@app.get("/api/widget/detect")
def detect_widget(lms_course_id: str, db: Session = Depends(get_db)):
    """
    Detect if a course bot exists for a given LMS course ID.
    Used by the floating widget loader script to auto-show widgets on matching pages.

    Returns course info if found, 404 if no matching course bot.
    """
    if not lms_course_id or len(lms_course_id) > 255:
        raise HTTPException(status_code=400, detail="Invalid LMS course ID")

    # Sanitize input
    lms_course_id = sanitize_string(lms_course_id.strip(), max_length=255)

    # Look up by LMS course ID
    course = db.query(Course).filter(Course.lms_course_id == lms_course_id).first()

    if not course:
        raise HTTPException(status_code=404, detail="No course bot found for this LMS course")

    # Use global widget style
    config = get_global_widget_style()

    return {
        "found": True,
        "course_id": course.id,
        "course_name": course.name,
        "lms_course_id": course.lms_course_id,
        "config": config
    }


# Widget config endpoint (for the embedded widget to fetch course info and styling)
@app.get("/api/widget/{course_id}")
def get_widget_config(course_id: str, db: Session = Depends(get_db)):
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    # Use global widget style
    config = get_global_widget_style()

    return {
        "course_id": course.id,
        "course_name": course.name,
        "course_description": course.description,
        "config": config
    }


def _course_to_response(course: Course) -> CourseResponse:
    # Generate embed code for this course
    embed_code = f'''<iframe
  src="YOUR_WIDGET_URL/widget.html?course_id={course.id}"
  width="400"
  height="600"
  frameborder="0"
  style="border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
</iframe>'''

    return CourseResponse(
        id=course.id,
        name=course.name,
        description=course.description,
        content=course.content,
        system_prompt=course.system_prompt,
        widget_config=course.widget_config,
        lms_course_id=course.lms_course_id,
        llm_provider=course.llm_provider,
        llm_model=course.llm_model,
        embed_code=embed_code
    )


# =============================================================================
# SETUP ENDPOINTS
# =============================================================================

class SetupRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8, max_length=100)


@app.get("/setup")
async def serve_setup():
    """Serve the setup page for first-time admin account creation."""
    if is_setup_complete():
        # Already set up, redirect to admin
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/", status_code=302)
    return FileResponse(PROJECT_ROOT / "frontend" / "setup.html")


@app.get("/api/setup/status")
async def get_setup_status():
    """Check if setup is complete."""
    return {"setup_complete": is_setup_complete()}


@app.post("/api/setup")
async def create_admin_account(setup: SetupRequest):
    """Create the admin account. Only works if no admin exists."""
    if is_setup_complete():
        raise HTTPException(
            status_code=400,
            detail="Setup already complete. Admin account already exists."
        )

    # Validate username
    if not re.match(r'^[a-zA-Z0-9_-]+$', setup.username):
        raise HTTPException(
            status_code=400,
            detail="Username can only contain letters, numbers, underscores, and hyphens."
        )

    db = SessionLocal()
    try:
        # Create admin user
        admin = AdminUser(username=setup.username)
        admin.set_password(setup.password)
        db.add(admin)
        db.commit()

        return {"message": "Admin account created successfully", "username": setup.username}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create admin account: {str(e)}")
    finally:
        db.close()


# =============================================================================
# SETTINGS API
# =============================================================================

class SettingsUpdate(BaseModel):
    """API Keys update - stores keys for each provider separately."""
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None


@app.get("/api/settings")
async def get_settings(request: Request):
    """Get current API key settings (requires authentication)."""
    credentials = await security(request)
    verify_admin_credentials(credentials)

    db = SessionLocal()
    try:
        settings = db.query(LLMSettings).filter(LLMSettings.id == "global").first()

        # Check which providers have API keys configured (database or environment)
        configured_providers = {
            "anthropic": bool(
                (settings and settings.anthropic_api_key) or
                os.getenv("ANTHROPIC_API_KEY")
            ),
            "openai": bool(
                (settings and settings.openai_api_key) or
                os.getenv("OPENAI_API_KEY")
            ),
            "google": bool(
                (settings and settings.google_api_key) or
                os.getenv("GOOGLE_API_KEY")
            ),
        }

        return {
            "configured_providers": configured_providers,
            "providers": LLM_PROVIDERS  # Send available providers/models for the course form
        }
    finally:
        db.close()


@app.post("/api/settings")
async def update_settings(request: Request, settings_update: SettingsUpdate):
    """Update API keys for LLM providers (requires authentication)."""
    credentials = await security(request)
    verify_admin_credentials(credentials)

    db = SessionLocal()
    try:
        settings = db.query(LLMSettings).filter(LLMSettings.id == "global").first()

        if not settings:
            # Create new settings
            settings = LLMSettings(id="global")
            db.add(settings)

        # Update API keys for each provider if provided
        if settings_update.anthropic_api_key is not None:
            settings.anthropic_api_key = settings_update.anthropic_api_key
        if settings_update.openai_api_key is not None:
            settings.openai_api_key = settings_update.openai_api_key
        if settings_update.google_api_key is not None:
            settings.google_api_key = settings_update.google_api_key

        db.commit()
        return {"message": "API keys saved successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save API keys: {str(e)}")
    finally:
        db.close()


# =============================================================================
# WIDGET STYLE API (Global Widget Appearance Settings)
# =============================================================================

class WidgetStyleUpdate(BaseModel):
    colors: Optional[dict] = None
    text: Optional[dict] = None
    style: Optional[dict] = None


@app.get("/api/widget-style")
async def get_widget_style(request: Request):
    """Get global widget style settings (requires authentication)."""
    credentials = await security(request)
    verify_admin_credentials(credentials)

    db = SessionLocal()
    try:
        style = db.query(WidgetStyle).filter(WidgetStyle.id == "global").first()
        if not style:
            # Return defaults if no settings exist
            return DEFAULT_WIDGET_CONFIG
        return style.config
    finally:
        db.close()


@app.post("/api/widget-style")
async def update_widget_style(request: Request, style_update: WidgetStyleUpdate):
    """Update global widget style settings (requires authentication)."""
    credentials = await security(request)
    verify_admin_credentials(credentials)

    db = SessionLocal()
    try:
        style = db.query(WidgetStyle).filter(WidgetStyle.id == "global").first()

        if not style:
            # Create new style record
            style = WidgetStyle(id="global", config=DEFAULT_WIDGET_CONFIG.copy())
            db.add(style)

        # Merge updates into existing config
        config = style.config.copy() if style.config else DEFAULT_WIDGET_CONFIG.copy()

        if style_update.colors:
            config["colors"] = {**config.get("colors", {}), **style_update.colors}
        if style_update.text:
            config["text"] = {**config.get("text", {}), **style_update.text}
        if style_update.style:
            config["style"] = {**config.get("style", {}), **style_update.style}

        style.config = config
        db.commit()
        return {"message": "Widget style saved successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save widget style: {str(e)}")
    finally:
        db.close()


def get_global_widget_style():
    """Get the current global widget style from database or defaults."""
    db = SessionLocal()
    try:
        style = db.query(WidgetStyle).filter(WidgetStyle.id == "global").first()
        if style and style.config:
            return style.config
        return DEFAULT_WIDGET_CONFIG
    finally:
        db.close()


# Serve static files
@app.get("/")
async def serve_admin(request: Request):
    """Serve admin dashboard, or redirect to setup if not configured."""
    if not is_setup_complete():
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/setup", status_code=302)

    # Require authentication
    credentials = await security(request)
    verify_admin_credentials(credentials)

    return FileResponse(PROJECT_ROOT / "frontend" / "admin.html")


@app.get("/admin")
async def serve_admin_alt(request: Request):
    """Serve admin dashboard, or redirect to setup if not configured."""
    if not is_setup_complete():
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/setup", status_code=302)

    # Require authentication
    credentials = await security(request)
    verify_admin_credentials(credentials)

    return FileResponse(PROJECT_ROOT / "frontend" / "admin.html")


@app.get("/widget.html")
async def serve_widget():
    return FileResponse(PROJECT_ROOT / "widget" / "widget.html")


@app.get("/widget-loader.js")
async def serve_widget_loader():
    """Serve the floating widget loader script."""
    return FileResponse(
        PROJECT_ROOT / "widget" / "widget-loader.js",
        media_type="application/javascript"
    )


# =============================================================================
# HEALTH CHECK & MONITORING
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring and load balancers."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "rate_limit_per_minute": RATE_LIMIT_PER_MINUTE
    }


@app.get("/api/status")
async def api_status(request: Request):
    """Get API status including rate limit info for the current client."""
    client_ip = get_client_ip(request)
    remaining = rate_limiter.get_remaining(client_ip)

    return {
        "status": "operational",
        "rate_limit": {
            "limit_per_minute": RATE_LIMIT_PER_MINUTE,
            "remaining": remaining
        }
    }


if __name__ == "__main__":
    import uvicorn

    # Get port from environment (for deployment platforms like Render)
    port = int(os.getenv("PORT", "8000"))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
