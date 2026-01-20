from sqlalchemy import create_engine, Column, String, Text, DateTime, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import uuid
import bcrypt
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./course_chat.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Default widget configuration
DEFAULT_WIDGET_CONFIG = {
    "colors": {
        "primary": "#0D2B3E",      # Header, user messages
        "accent": "#5FCFB4",        # Icons, send button
        "background": "#F8FAFC",    # Chat area background
        "messageBg": "#FFFFFF",     # Assistant message background
        "messageText": "#334155",   # Message text color
    },
    "text": {
        "title": "Course Assistant",
        "subtitle": "Ask questions about this course",
        "placeholder": "Type your question...",
        "welcomeMessage": "Welcome! I'm here to help answer any questions you have about this course. What would you like to know?"
    },
    "style": {
        "borderRadius": "8",        # px - for corners
        "iconStyle": "minimal",     # minimal, rounded, square
        "widgetTheme": "light",     # light, dark
        "position": "bottom-right", # bottom-right, bottom-left
    }
}


class AdminUser(Base):
    """Admin user for the dashboard. Only one admin is supported."""
    __tablename__ = "admin_users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(255), nullable=False, unique=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    def set_password(self, password: str):
        """Hash and set the password."""
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def verify_password(self, password: str) -> bool:
        """Verify a password against the hash."""
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))


class LLMSettings(Base):
    """Global LLM settings - stores API keys for all providers."""
    __tablename__ = "llm_settings"

    id = Column(String, primary_key=True, default="global")
    provider = Column(String(50), nullable=False, default="anthropic")  # Default provider (legacy)
    model = Column(String(100), nullable=False, default="claude-sonnet-4-20250514")  # Default model (legacy)
    api_key_encrypted = Column(Text, nullable=True)  # Legacy single API key
    anthropic_api_key = Column(Text, nullable=True)  # Anthropic API key
    openai_api_key = Column(Text, nullable=True)  # OpenAI API key
    google_api_key = Column(Text, nullable=True)  # Google API key
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class WidgetStyle(Base):
    """Global widget style settings - only one row exists."""
    __tablename__ = "widget_style"

    id = Column(String, primary_key=True, default="global")
    config = Column(JSON, nullable=False, default=DEFAULT_WIDGET_CONFIG)  # Full widget config (colors, text, style)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Course(Base):
    __tablename__ = "courses"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    content = Column(Text, nullable=False)  # The course content/transcript
    system_prompt = Column(Text, nullable=True)  # Optional custom system prompt
    widget_config = Column(JSON, nullable=True, default=DEFAULT_WIDGET_CONFIG)  # Widget customization
    lms_course_id = Column(String(255), nullable=True, index=True)  # External LMS course ID (Skilljar, etc.)
    llm_provider = Column(String(50), nullable=True)  # anthropic, openai, google - null means use global default
    llm_model = Column(String(100), nullable=True)  # specific model ID - null means use provider default
    is_active = Column(Boolean, nullable=False, default=True)  # Whether the course bot is active
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)
