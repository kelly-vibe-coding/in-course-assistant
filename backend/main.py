from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session
from typing import Optional, AsyncGenerator, Literal
from datetime import datetime
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

from database import get_db, init_db, Course, AdminUser, LLMSettings, WidgetStyle, ChatLog, QuestionCluster, SessionLocal, DEFAULT_WIDGET_CONFIG

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
    Returns the email if valid, raises 401 if not.
    Also updates last_login_at timestamp.
    """
    db = SessionLocal()
    try:
        admin = db.query(AdminUser).filter(AdminUser.email == credentials.username).first()

        if not admin or not admin.verify_password(credentials.password):
            raise HTTPException(
                status_code=401,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Basic realm=\"Admin Access\""}
            )

        # Update last login timestamp
        admin.last_login_at = datetime.utcnow()
        db.commit()

        return credentials.username
    finally:
        db.close()


def get_current_user(credentials: HTTPBasicCredentials = Depends(security)) -> AdminUser:
    """
    Get the current authenticated user object.
    Returns the full AdminUser if valid, raises 401 if not.
    """
    db = SessionLocal()
    try:
        user = db.query(AdminUser).filter(AdminUser.email == credentials.username).first()

        if not user or not user.verify_password(credentials.password):
            raise HTTPException(
                status_code=401,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Basic realm=\"Admin Access\""}
            )

        # Detach user from session so it can be used after session closes
        db.expunge(user)
        return user
    finally:
        db.close()


def require_first_user(credentials: HTTPBasicCredentials = Depends(security)) -> AdminUser:
    """
    Dependency that requires the first user (admin) for user management.
    Only the first user created can add/remove other users.
    """
    user = get_current_user(credentials)
    if not user.is_first_user:
        raise HTTPException(
            status_code=403,
            detail="Only the admin can manage users"
        )
    return user

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
# CHAT LOGGING & GROUNDING DETECTION
# =============================================================================

def parse_grounding_metadata(response: str) -> tuple:
    """
    Parse grounding metadata from AI response.
    Returns: (clean_response, grounding_source, is_content_gap, gap_topic)
    """
    # Pattern to match the metadata line: [[GROUNDING:source|gap_detected|topic]]
    pattern = r'\[\[GROUNDING:(\w+)\|(true|false)\|([^\]]+)\]\]'
    match = re.search(pattern, response)

    if match:
        source = match.group(1)  # course_content, general_knowledge, mixed, uncertain
        gap_detected = match.group(2) == "true"
        topic = match.group(3) if match.group(3) != "none" else None

        # Remove the metadata line from response
        clean_response = re.sub(r'\n?\[\[GROUNDING:[^\]]+\]\]', '', response).strip()

        return clean_response, source, gap_detected, topic

    # Fallback detection if LLM didn't include metadata
    # Check for known uncertainty phrases
    uncertainty_phrases = [
        "isn't specifically covered",
        "not specifically covered",
        "this isn't covered",
        "not covered in the course",
        "beyond what's in the course"
    ]
    is_uncertain = any(phrase in response.lower() for phrase in uncertainty_phrases)

    return response, "unknown", is_uncertain, None


def get_analytics_logging_level(db: Session) -> str:
    """Get the current analytics logging level from settings."""
    settings = db.query(LLMSettings).filter(LLMSettings.id == "global").first()
    if settings and settings.analytics_logging_level:
        return settings.analytics_logging_level
    return "analytics_only"  # Default to privacy-preserving mode


def log_chat_interaction(
    db: Session,
    course_id: str,
    session_id: str,
    lesson_title: str,
    user_message: str,
    ai_response: str,
    grounding_source: str,
    is_content_gap: bool,
    gap_topic: str,
    response_time_ms: int,
    provider: str,
    model: str
):
    """Log a chat interaction to the database for analytics.

    Respects the analytics_logging_level setting:
    - "full": Logs everything including message content
    - "analytics_only": Logs metadata (grounding, gaps) but not message content
    - "disabled": No logging at all
    """
    import uuid

    logging_level = get_analytics_logging_level(db)

    if logging_level == "disabled":
        return  # Don't log anything

    # For analytics_only mode, redact the actual message content
    if logging_level == "analytics_only":
        user_message = "[Content not logged - analytics only mode]"
        ai_response = "[Content not logged - analytics only mode]"

    chat_log = ChatLog(
        id=str(uuid.uuid4()),
        session_id=session_id or str(uuid.uuid4()),
        course_id=course_id,
        lesson_title=lesson_title,
        user_message=user_message,
        ai_response=ai_response,
        grounding_source=grounding_source,
        is_content_gap=is_content_gap,
        gap_topic=gap_topic,
        response_time_ms=response_time_ms,
        llm_provider=provider,
        llm_model=model
    )
    db.add(chat_log)
    db.commit()


# Grounding tagging instructions to append to system prompts
GROUNDING_TAGGING_INSTRUCTIONS = """

=== RESPONSE GROUNDING TAGGING (INTERNAL) ===
After your response, you MUST include a metadata line on the LAST LINE in exactly this format:
[[GROUNDING:source|gap_detected|topic]]

Where:
- source: "course_content" (answer fully from course), "general_knowledge" (answer from your knowledge), "mixed" (combination), or "uncertain" (couldn't fully answer)
- gap_detected: "true" if the question reveals missing course content that SHOULD be there, "false" otherwise
- topic: brief topic of any content gap (or "none" if no gap)

Examples:
- [[GROUNDING:course_content|false|none]]
- [[GROUNDING:mixed|false|none]]
- [[GROUNDING:uncertain|true|API authentication methods]]
- [[GROUNDING:general_knowledge|true|industry best practices for testing]]

This metadata helps improve the course. The user will never see this line."""

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
    is_active: bool = True  # Whether the course bot is active

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
    is_active: Optional[bool] = None  # Whether the course bot is active

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
    is_active: bool
    embed_code: str

    class Config:
        from_attributes = True


class ChatRequest(BaseModel):
    course_id: str = Field(..., min_length=36, max_length=36)
    message: str = Field(..., min_length=1)
    conversation_history: Optional[list] = []
    lesson_title: Optional[str] = Field(None, max_length=500)  # Current lesson context
    complexity: Optional[str] = Field("detailed", pattern="^(simple|detailed)$")  # Response complexity level
    session_id: Optional[str] = Field(None, max_length=64)  # Session ID for conversation grouping

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

    @validator('lesson_title')
    def sanitize_lesson_title(cls, v):
        if v:
            return sanitize_string(v, max_length=500)
        return v

    @validator('session_id')
    def sanitize_session_id(cls, v):
        if v:
            return sanitize_string(v, max_length=64)
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
        llm_model=course.llm_model,
        is_active=course.is_active
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
    if course_update.is_active is not None:
        course.is_active = course_update.is_active

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


@app.post("/api/courses/{course_id}/toggle")
def toggle_course_active(course_id: str, db: Session = Depends(get_db), username: str = Depends(verify_admin_credentials)):
    """Toggle a course's active status."""
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    # Toggle the is_active status
    course.is_active = not course.is_active
    db.commit()
    db.refresh(course)

    return {
        "message": f"Course {'activated' if course.is_active else 'deactivated'} successfully",
        "is_active": course.is_active
    }


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

    # Start timing for response_time_ms
    start_time = time.time()

    course = db.query(Course).filter(Course.id == request.course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    # Get LLM settings: use course-specific if set, otherwise fall back to global
    if course.llm_provider and course.llm_model:
        provider = course.llm_provider
        model = course.llm_model
    else:
        provider, model = get_configured_provider_and_model()

    # Build complexity instructions based on user preference
    complexity = getattr(request, 'complexity', 'detailed') or 'detailed'
    complexity_instructions = ""
    if complexity == "simple":
        complexity_instructions = """
RESPONSE STYLE: Keep your responses CONCISE and SIMPLE.
- Use short sentences and simple language
- Focus on the key point without elaboration
- Avoid jargon unless necessary
- Aim for responses that are easy to quickly scan and understand
- Use bullet points when listing information"""
    else:
        complexity_instructions = """
RESPONSE STYLE: Provide DETAILED and THOROUGH responses.
- Explain concepts fully with context and examples
- Use proper terminology and explain it when needed
- Connect ideas to help build deeper understanding
- Include relevant details that enrich the explanation"""

    # Build lesson context if provided
    lesson_title = getattr(request, 'lesson_title', None)
    lesson_context = ""
    if lesson_title:
        lesson_context = f"""
CURRENT LESSON CONTEXT: The learner is currently studying "{lesson_title}".
- Prioritize information relevant to this specific lesson
- When answering questions, consider what they're learning in this lesson
- If they ask to be quizzed, focus questions on this lesson's content"""

    # Build the system prompt - Smart prompt that stays on-topic but allows helpful context
    base_system = course.system_prompt or f"""You are a helpful learning assistant for the course "{course.name}".

PRIMARY ROLE: Help learners understand and succeed in this course.
{complexity_instructions}
{lesson_context}

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

6. QUIZ MODE - When the learner asks to be quizzed:
   - Ask one question at a time
   - Wait for their answer before giving feedback
   - Provide encouraging feedback and explanations
   - Keep track of their progress in the conversation

COURSE NAME: {course.name}"""

    system_prompt = f"""{base_system}

=== COURSE CONTENT ===
{course.content}
=== END COURSE CONTENT ==={GROUNDING_TAGGING_INSTRUCTIONS}"""

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

        # Parse grounding metadata and get clean response
        clean_response, grounding_source, is_content_gap, gap_topic = parse_grounding_metadata(response_text)

        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)

        # Log the interaction for analytics
        try:
            log_chat_interaction(
                db=db,
                course_id=request.course_id,
                session_id=getattr(request, 'session_id', None),
                lesson_title=lesson_title,
                user_message=request.message,
                ai_response=clean_response,
                grounding_source=grounding_source,
                is_content_gap=is_content_gap,
                gap_topic=gap_topic,
                response_time_ms=response_time_ms,
                provider=provider,
                model=model
            )
        except Exception as log_error:
            # Don't fail the request if logging fails
            print(f"Warning: Failed to log chat interaction: {log_error}")

        return ChatResponse(response=clean_response)

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

    # Start timing for response_time_ms
    start_time = time.time()

    course = db.query(Course).filter(Course.id == request.course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    # Get LLM settings: use course-specific if set, otherwise fall back to global
    if course.llm_provider and course.llm_model:
        provider = course.llm_provider
        model = course.llm_model
    else:
        provider, model = get_configured_provider_and_model()

    # Build complexity instructions based on user preference
    complexity = getattr(request, 'complexity', 'detailed') or 'detailed'
    complexity_instructions = ""
    if complexity == "simple":
        complexity_instructions = """
RESPONSE STYLE: Keep your responses CONCISE and SIMPLE.
- Use short sentences and simple language
- Focus on the key point without elaboration
- Avoid jargon unless necessary
- Aim for responses that are easy to quickly scan and understand
- Use bullet points when listing information"""
    else:
        complexity_instructions = """
RESPONSE STYLE: Provide DETAILED and THOROUGH responses.
- Explain concepts fully with context and examples
- Use proper terminology and explain it when needed
- Connect ideas to help build deeper understanding
- Include relevant details that enrich the explanation"""

    # Build lesson context if provided
    lesson_title = getattr(request, 'lesson_title', None)
    lesson_context = ""
    if lesson_title:
        lesson_context = f"""
CURRENT LESSON CONTEXT: The learner is currently studying "{lesson_title}".
- Prioritize information relevant to this specific lesson
- When answering questions, consider what they're learning in this lesson
- If they ask to be quizzed, focus questions on this lesson's content"""

    # Build the system prompt
    base_system = course.system_prompt or f"""You are a helpful learning assistant for the course "{course.name}".

PRIMARY ROLE: Help learners understand and succeed in this course.
{complexity_instructions}
{lesson_context}

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

6. QUIZ MODE - When the learner asks to be quizzed:
   - Ask one question at a time
   - Wait for their answer before giving feedback
   - Provide encouraging feedback and explanations
   - Keep track of their progress in the conversation

COURSE NAME: {course.name}"""

    system_prompt = f"""{base_system}

=== COURSE CONTENT ===
{course.content}
=== END COURSE CONTENT ==={GROUNDING_TAGGING_INSTRUCTIONS}"""

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

    # Capture request context for logging after stream completes
    request_context = {
        "course_id": request.course_id,
        "session_id": getattr(request, 'session_id', None),
        "lesson_title": lesson_title,
        "user_message": request.message,
        "provider": provider,
        "model": model,
        "start_time": start_time
    }

    async def generate_anthropic():
        """Stream from Anthropic Claude."""
        api_key = get_api_key_for_provider("anthropic")
        if not api_key:
            yield f"data: {json.dumps({'error': 'Anthropic API key not configured'})}\n\n"
            return

        client = anthropic.Anthropic(api_key=api_key)
        full_response = ""
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
                    full_response += text
                    # Don't stream the grounding metadata to the client
                    if "[[GROUNDING:" not in full_response:
                        yield f"data: {json.dumps({'text': text})}\n\n"
                    elif "[[GROUNDING:" in text:
                        # We've hit the metadata, stop streaming but continue collecting
                        pass

            # Parse grounding metadata and log
            clean_response, grounding_source, is_content_gap, gap_topic = parse_grounding_metadata(full_response)
            response_time_ms = int((time.time() - request_context["start_time"]) * 1000)

            # Log the interaction
            try:
                log_db = SessionLocal()
                log_chat_interaction(
                    db=log_db,
                    course_id=request_context["course_id"],
                    session_id=request_context["session_id"],
                    lesson_title=request_context["lesson_title"],
                    user_message=request_context["user_message"],
                    ai_response=clean_response,
                    grounding_source=grounding_source,
                    is_content_gap=is_content_gap,
                    gap_topic=gap_topic,
                    response_time_ms=response_time_ms,
                    provider=request_context["provider"],
                    model=request_context["model"]
                )
                log_db.close()
            except Exception as log_error:
                print(f"Warning: Failed to log streaming chat interaction: {log_error}")

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

        full_response = ""
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=openai_messages,
                max_tokens=1024,
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    full_response += text
                    # Don't stream the grounding metadata to the client
                    if "[[GROUNDING:" not in full_response:
                        yield f"data: {json.dumps({'text': text})}\n\n"

            # Parse grounding metadata and log
            clean_response, grounding_source, is_content_gap, gap_topic = parse_grounding_metadata(full_response)
            response_time_ms = int((time.time() - request_context["start_time"]) * 1000)

            # Log the interaction
            try:
                log_db = SessionLocal()
                log_chat_interaction(
                    db=log_db,
                    course_id=request_context["course_id"],
                    session_id=request_context["session_id"],
                    lesson_title=request_context["lesson_title"],
                    user_message=request_context["user_message"],
                    ai_response=clean_response,
                    grounding_source=grounding_source,
                    is_content_gap=is_content_gap,
                    gap_topic=gap_topic,
                    response_time_ms=response_time_ms,
                    provider=request_context["provider"],
                    model=request_context["model"]
                )
                log_db.close()
            except Exception as log_error:
                print(f"Warning: Failed to log streaming chat interaction: {log_error}")

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

        full_response = ""
        try:
            response = chat.send_message(current_message, stream=True)
            for chunk in response:
                if chunk.text:
                    text = chunk.text
                    full_response += text
                    # Don't stream the grounding metadata to the client
                    if "[[GROUNDING:" not in full_response:
                        yield f"data: {json.dumps({'text': text})}\n\n"

            # Parse grounding metadata and log
            clean_response, grounding_source, is_content_gap, gap_topic = parse_grounding_metadata(full_response)
            response_time_ms = int((time.time() - request_context["start_time"]) * 1000)

            # Log the interaction
            try:
                log_db = SessionLocal()
                log_chat_interaction(
                    db=log_db,
                    course_id=request_context["course_id"],
                    session_id=request_context["session_id"],
                    lesson_title=request_context["lesson_title"],
                    user_message=request_context["user_message"],
                    ai_response=clean_response,
                    grounding_source=grounding_source,
                    is_content_gap=is_content_gap,
                    gap_topic=gap_topic,
                    response_time_ms=response_time_ms,
                    provider=request_context["provider"],
                    model=request_context["model"]
                )
                log_db.close()
            except Exception as log_error:
                print(f"Warning: Failed to log streaming chat interaction: {log_error}")

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
# ANALYTICS API ENDPOINTS
# =============================================================================

from datetime import timedelta
from sqlalchemy import func, desc, Integer as SQLInteger

@app.get("/api/analytics/overview")
def get_analytics_overview(
    course_id: Optional[str] = None,
    days: int = 30,
    db: Session = Depends(get_db),
    username: str = Depends(verify_admin_credentials)
):
    """Get high-level analytics summary."""
    cutoff_date = datetime.utcnow() - timedelta(days=days)

    # Base query
    query = db.query(ChatLog).filter(ChatLog.created_at >= cutoff_date)
    if course_id:
        query = query.filter(ChatLog.course_id == course_id)

    # Total conversations
    total_chats = query.count()

    # Content gaps count
    content_gaps = query.filter(ChatLog.is_content_gap == True).count()

    # Grounding distribution
    grounding_stats = db.query(
        ChatLog.grounding_source,
        func.count(ChatLog.id).label('count')
    ).filter(ChatLog.created_at >= cutoff_date)
    if course_id:
        grounding_stats = grounding_stats.filter(ChatLog.course_id == course_id)
    grounding_stats = grounding_stats.group_by(ChatLog.grounding_source).all()

    grounding_distribution = {stat.grounding_source: stat.count for stat in grounding_stats}

    # Course-grounded percentage
    course_grounded = grounding_distribution.get('course_content', 0) + grounding_distribution.get('mixed', 0)
    grounded_percentage = round((course_grounded / total_chats * 100) if total_chats > 0 else 0, 1)

    # Unique lessons with questions
    lessons_query = db.query(func.count(func.distinct(ChatLog.lesson_title))).filter(
        ChatLog.created_at >= cutoff_date,
        ChatLog.lesson_title.isnot(None)
    )
    if course_id:
        lessons_query = lessons_query.filter(ChatLog.course_id == course_id)
    active_lessons = lessons_query.scalar() or 0

    # Average response time
    avg_response_time = db.query(func.avg(ChatLog.response_time_ms)).filter(
        ChatLog.created_at >= cutoff_date
    )
    if course_id:
        avg_response_time = avg_response_time.filter(ChatLog.course_id == course_id)
    avg_response_time = avg_response_time.scalar() or 0

    return {
        "total_chats": total_chats,
        "content_gaps": content_gaps,
        "grounded_percentage": grounded_percentage,
        "active_lessons": active_lessons,
        "avg_response_time_ms": round(avg_response_time),
        "grounding_distribution": grounding_distribution,
        "period_days": days
    }


@app.get("/api/analytics/content-gaps")
def get_content_gaps(
    course_id: Optional[str] = None,
    days: int = 30,
    limit: int = 50,
    db: Session = Depends(get_db),
    username: str = Depends(verify_admin_credentials)
):
    """Get questions the AI couldn't answer from course content."""
    cutoff_date = datetime.utcnow() - timedelta(days=days)

    query = db.query(ChatLog).filter(
        ChatLog.created_at >= cutoff_date,
        ChatLog.is_content_gap == True
    )
    if course_id:
        query = query.filter(ChatLog.course_id == course_id)

    gaps = query.order_by(desc(ChatLog.created_at)).limit(limit).all()

    # Group by topic if available
    topic_counts = {}
    for gap in gaps:
        topic = gap.gap_topic or "Uncategorized"
        if topic not in topic_counts:
            topic_counts[topic] = {"count": 0, "questions": [], "lessons": set()}
        topic_counts[topic]["count"] += 1
        topic_counts[topic]["questions"].append(gap.user_message)
        if gap.lesson_title:
            topic_counts[topic]["lessons"].add(gap.lesson_title)

    # Convert to list and sort by count
    gap_topics = [
        {
            "topic": topic,
            "count": data["count"],
            "sample_questions": data["questions"][:3],
            "lessons": list(data["lessons"])[:3]
        }
        for topic, data in topic_counts.items()
    ]
    gap_topics.sort(key=lambda x: x["count"], reverse=True)

    return {
        "total_gaps": len(gaps),
        "gap_topics": gap_topics[:20],
        "recent_gaps": [
            {
                "id": gap.id,
                "question": gap.user_message,
                "topic": gap.gap_topic,
                "lesson": gap.lesson_title,
                "created_at": gap.created_at.isoformat()
            }
            for gap in gaps[:10]
        ]
    }


@app.get("/api/analytics/lesson-friction")
def get_lesson_friction(
    course_id: str,
    days: int = 30,
    db: Session = Depends(get_db),
    username: str = Depends(verify_admin_credentials)
):
    """Get question volume by lesson (friction heatmap data)."""
    cutoff_date = datetime.utcnow() - timedelta(days=days)

    # Query for lessons with question counts
    lesson_stats = db.query(
        ChatLog.lesson_title,
        func.count(ChatLog.id).label('total_questions'),
        func.sum(func.cast(ChatLog.is_content_gap, SQLInteger)).label('gap_count')
    ).filter(
        ChatLog.course_id == course_id,
        ChatLog.created_at >= cutoff_date,
        ChatLog.lesson_title.isnot(None)
    ).group_by(ChatLog.lesson_title).order_by(desc('total_questions')).all()

    lessons = []
    max_questions = max((stat.total_questions for stat in lesson_stats), default=1)

    for stat in lesson_stats:
        gap_count = stat.gap_count or 0
        gap_percentage = round((gap_count / stat.total_questions * 100) if stat.total_questions > 0 else 0, 1)
        lessons.append({
            "lesson": stat.lesson_title,
            "question_count": stat.total_questions,
            "gap_count": gap_count,
            "gap_percentage": gap_percentage,
            "heat_score": round(stat.total_questions / max_questions * 100)
        })

    return {
        "lessons": lessons,
        "total_lessons": len(lessons),
        "period_days": days
    }


@app.get("/api/analytics/conversations")
def get_conversations(
    course_id: Optional[str] = None,
    days: int = 30,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    username: str = Depends(verify_admin_credentials)
):
    """Get conversation sessions for review."""
    cutoff_date = datetime.utcnow() - timedelta(days=days)

    # Get unique sessions with metadata
    session_query = db.query(
        ChatLog.session_id,
        ChatLog.course_id,
        func.min(ChatLog.created_at).label('started_at'),
        func.max(ChatLog.created_at).label('last_message_at'),
        func.count(ChatLog.id).label('message_count'),
        func.sum(func.cast(ChatLog.is_content_gap, SQLInteger)).label('gap_count')
    ).filter(ChatLog.created_at >= cutoff_date)

    if course_id:
        session_query = session_query.filter(ChatLog.course_id == course_id)

    sessions = session_query.group_by(
        ChatLog.session_id, ChatLog.course_id
    ).order_by(desc('last_message_at')).offset(offset).limit(limit).all()

    # Get course names for context
    course_ids = list(set(s.course_id for s in sessions))
    courses = {c.id: c.name for c in db.query(Course).filter(Course.id.in_(course_ids)).all()}

    return {
        "sessions": [
            {
                "session_id": s.session_id,
                "course_id": s.course_id,
                "course_name": courses.get(s.course_id, "Unknown"),
                "started_at": s.started_at.isoformat() if s.started_at else None,
                "last_message_at": s.last_message_at.isoformat() if s.last_message_at else None,
                "message_count": s.message_count,
                "has_gaps": (s.gap_count or 0) > 0
            }
            for s in sessions
        ],
        "total": len(sessions),
        "offset": offset,
        "limit": limit
    }


@app.get("/api/analytics/conversation/{session_id}")
def get_conversation_detail(
    session_id: str,
    db: Session = Depends(get_db),
    username: str = Depends(verify_admin_credentials)
):
    """Get full conversation thread for a session."""
    messages = db.query(ChatLog).filter(
        ChatLog.session_id == session_id
    ).order_by(ChatLog.created_at).all()

    if not messages:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Get course info
    course = db.query(Course).filter(Course.id == messages[0].course_id).first()

    return {
        "session_id": session_id,
        "course_id": messages[0].course_id,
        "course_name": course.name if course else "Unknown",
        "messages": [
            {
                "id": msg.id,
                "user_message": msg.user_message,
                "ai_response": msg.ai_response,
                "lesson_title": msg.lesson_title,
                "grounding_source": msg.grounding_source,
                "is_content_gap": msg.is_content_gap,
                "gap_topic": msg.gap_topic,
                "response_time_ms": msg.response_time_ms,
                "created_at": msg.created_at.isoformat()
            }
            for msg in messages
        ],
        "total_messages": len(messages)
    }


@app.post("/api/analytics/analyze-questions")
def analyze_questions(
    course_id: str,
    days: int = 30,
    db: Session = Depends(get_db),
    username: str = Depends(verify_admin_credentials)
):
    """AI-assisted analysis: cluster questions into themes and generate FAQ drafts."""
    cutoff_date = datetime.utcnow() - timedelta(days=days)

    # Get recent chat logs for this course
    chat_logs = db.query(ChatLog).filter(
        ChatLog.course_id == course_id,
        ChatLog.created_at >= cutoff_date
    ).all()

    if len(chat_logs) < 5:
        raise HTTPException(
            status_code=400,
            detail="Not enough data to analyze (minimum 5 conversations required)"
        )

    # Prepare questions for analysis
    questions = [
        {
            "question": log.user_message,
            "lesson": log.lesson_title,
            "was_gap": log.is_content_gap
        }
        for log in chat_logs
    ]

    # Get LLM to cluster and analyze
    provider, model = get_configured_provider_and_model()

    analysis_prompt = f"""Analyze these {len(questions)} learner questions from a course.

1. Group them into 3-7 themes based on what learners are asking about
2. For each theme, identify:
   - A clear theme name (2-5 words)
   - A brief description
   - 2-3 representative sample questions
   - Whether this theme represents a content gap (learners confused about something not well covered)
   - A suggested FAQ answer that could be added to the course

Return JSON in this exact format:
{{
    "themes": [
        {{
            "name": "Theme Name",
            "description": "Brief description",
            "question_count": 12,
            "sample_questions": ["Q1", "Q2", "Q3"],
            "is_content_gap": true,
            "suggested_faq": "Suggested answer text..."
        }}
    ],
    "overall_insights": "Brief summary of what learners struggle with most"
}}

Questions to analyze:
{json.dumps(questions[:100], indent=2)}"""

    system = "You are an instructional design analyst. Analyze learner questions to identify patterns and content gaps. Return only valid JSON."

    try:
        response_text = get_llm_response(provider, model, [{"role": "user", "content": analysis_prompt}], system, stream=False)

        # Handle potential markdown code blocks
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()

        analysis = json.loads(response_text)

        # Clear old clusters for this course
        db.query(QuestionCluster).filter(QuestionCluster.course_id == course_id).delete()

        # Create new clusters
        for theme in analysis.get("themes", []):
            cluster = QuestionCluster(
                course_id=course_id,
                theme=theme["name"],
                description=theme.get("description"),
                question_count=theme.get("question_count", 0),
                sample_questions=theme.get("sample_questions", []),
                suggested_faq_answer=theme.get("suggested_faq")
            )
            db.add(cluster)

        db.commit()

        return {
            "clusters_created": len(analysis.get("themes", [])),
            "overall_insights": analysis.get("overall_insights"),
            "themes": analysis.get("themes", [])
        }

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse AI analysis response")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/api/analytics/clusters/{course_id}")
def get_question_clusters(
    course_id: str,
    db: Session = Depends(get_db),
    username: str = Depends(verify_admin_credentials)
):
    """Get AI-generated question clusters for a course."""
    clusters = db.query(QuestionCluster).filter(
        QuestionCluster.course_id == course_id
    ).order_by(desc(QuestionCluster.question_count)).all()

    return {
        "clusters": [
            {
                "id": c.id,
                "theme": c.theme,
                "description": c.description,
                "question_count": c.question_count,
                "sample_questions": c.sample_questions,
                "suggested_faq_answer": c.suggested_faq_answer,
                "last_analyzed_at": c.last_analyzed_at.isoformat() if c.last_analyzed_at else None
            }
            for c in clusters
        ],
        "total": len(clusters)
    }


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

    Returns course info if found and active, 404 if no matching course bot or inactive.
    """
    if not lms_course_id or len(lms_course_id) > 255:
        raise HTTPException(status_code=400, detail="Invalid LMS course ID")

    # Sanitize input
    lms_course_id = sanitize_string(lms_course_id.strip(), max_length=255)

    # Look up by LMS course ID - only return active courses
    course = db.query(Course).filter(
        Course.lms_course_id == lms_course_id,
        Course.is_active == True
    ).first()

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
        is_active=course.is_active if course.is_active is not None else True,
        embed_code=embed_code
    )


# =============================================================================
# SETUP ENDPOINTS
# =============================================================================

class SetupRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=255)
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

    # Validate email format
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, setup.email):
        raise HTTPException(
            status_code=400,
            detail="Please enter a valid email address."
        )

    db = SessionLocal()
    try:
        # Create admin user - first user gets admin privileges
        admin = AdminUser(email=setup.email, is_first_user=True)
        admin.set_password(setup.password)
        db.add(admin)
        db.commit()

        return {"message": "Admin account created successfully", "email": setup.email}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create admin account: {str(e)}")
    finally:
        db.close()


# =============================================================================
# USER MANAGEMENT API
# =============================================================================

class UserCreate(BaseModel):
    email: str = Field(..., min_length=5, max_length=255)
    password: str = Field(..., min_length=8, max_length=100)


class UserResponse(BaseModel):
    id: str
    email: str
    is_first_user: bool
    created_at: datetime

    class Config:
        from_attributes = True


class PasswordChange(BaseModel):
    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=8, max_length=100)


@app.get("/api/users/me")
async def get_current_user_info(request: Request):
    """Get info about the current logged-in user."""
    credentials = await security(request)
    user = get_current_user(credentials)

    return {
        "id": user.id,
        "email": user.email,
        "is_first_user": user.is_first_user,
        "created_at": user.created_at.isoformat()
    }


@app.put("/api/users/me/password")
async def change_own_password(request: Request, password_change: PasswordChange):
    """Change the current user's password."""
    credentials = await security(request)
    user = get_current_user(credentials)

    # Verify current password
    if not user.verify_password(password_change.current_password):
        raise HTTPException(
            status_code=400,
            detail="Current password is incorrect"
        )

    db = SessionLocal()
    try:
        # Get fresh user from DB
        db_user = db.query(AdminUser).filter(AdminUser.id == user.id).first()
        if not db_user:
            raise HTTPException(status_code=404, detail="User not found")

        db_user.set_password(password_change.new_password)
        db.commit()

        return {"message": "Password changed successfully"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to change password: {str(e)}")
    finally:
        db.close()


@app.get("/api/users")
async def list_users(request: Request):
    """List all users. Only accessible by the first user (admin)."""
    credentials = await security(request)
    require_first_user(credentials)

    db = SessionLocal()
    try:
        users = db.query(AdminUser).order_by(AdminUser.created_at).all()
        return [
            {
                "id": u.id,
                "email": u.email,
                "is_first_user": u.is_first_user,
                "created_at": u.created_at.isoformat(),
                "last_login_at": u.last_login_at.isoformat() if u.last_login_at else None
            }
            for u in users
        ]
    finally:
        db.close()


@app.post("/api/users")
async def create_user(request: Request, user_create: UserCreate):
    """Create a new user. Only accessible by the first user (admin)."""
    credentials = await security(request)
    require_first_user(credentials)

    # Validate email format
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, user_create.email):
        raise HTTPException(
            status_code=400,
            detail="Please enter a valid email address."
        )

    db = SessionLocal()
    try:
        # Check if email already exists
        existing = db.query(AdminUser).filter(AdminUser.email == user_create.email).first()
        if existing:
            raise HTTPException(
                status_code=400,
                detail="A user with this email already exists."
            )

        # Create new user (not first user, so no admin privileges)
        new_user = AdminUser(email=user_create.email, is_first_user=False)
        new_user.set_password(user_create.password)
        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        return {
            "id": new_user.id,
            "email": new_user.email,
            "is_first_user": new_user.is_first_user,
            "created_at": new_user.created_at.isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create user: {str(e)}")
    finally:
        db.close()


@app.delete("/api/users/{user_id}")
async def delete_user(request: Request, user_id: str):
    """Delete a user. Only accessible by the first user (admin). Cannot delete self."""
    credentials = await security(request)
    admin = require_first_user(credentials)

    # Prevent self-deletion
    if user_id == admin.id:
        raise HTTPException(
            status_code=400,
            detail="You cannot delete your own account."
        )

    db = SessionLocal()
    try:
        user = db.query(AdminUser).filter(AdminUser.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        db.delete(user)
        db.commit()

        return {"message": "User deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete user: {str(e)}")
    finally:
        db.close()


# =============================================================================
# SETTINGS API
# =============================================================================

class SettingsUpdate(BaseModel):
    """API Keys and settings update."""
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    analytics_logging_level: Optional[str] = None  # "full", "analytics_only", "disabled"


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
            "providers": LLM_PROVIDERS,  # Send available providers/models for the course form
            "analytics_logging_level": settings.analytics_logging_level if settings else "analytics_only"
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

        # Update analytics logging level if provided
        if settings_update.analytics_logging_level is not None:
            if settings_update.analytics_logging_level in ["full", "analytics_only", "disabled"]:
                settings.analytics_logging_level = settings_update.analytics_logging_level
            else:
                raise HTTPException(status_code=400, detail="Invalid analytics_logging_level. Must be 'full', 'analytics_only', or 'disabled'")

        db.commit()
        return {"message": "Settings saved successfully"}
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
