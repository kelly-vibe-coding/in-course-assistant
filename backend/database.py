from sqlalchemy import create_engine, Column, String, Text, DateTime, JSON, Boolean, Integer, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
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
    """Admin user for the dashboard. Multiple users supported."""
    __tablename__ = "admin_users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), nullable=False, unique=True)
    password_hash = Column(String(255), nullable=False)
    is_first_user = Column(Boolean, default=False)  # Only first user can manage other users
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
    # Analytics logging level: "full" (log everything), "analytics_only" (metadata only, no messages), "disabled" (no logging)
    analytics_logging_level = Column(String(20), nullable=False, default="analytics_only")
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

    # Relationships for analytics
    chat_logs = relationship("ChatLog", back_populates="course", cascade="all, delete-orphan")
    question_clusters = relationship("QuestionCluster", back_populates="course", cascade="all, delete-orphan")


class ChatLog(Base):
    """Stores all chat interactions for analytics and content gap detection."""
    __tablename__ = "chat_logs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(64), nullable=False, index=True)  # Groups conversations
    course_id = Column(String(36), ForeignKey('courses.id'), nullable=False, index=True)
    lesson_title = Column(String(500), nullable=True, index=True)  # Current lesson context

    # Message content
    user_message = Column(Text, nullable=False)
    ai_response = Column(Text, nullable=False)

    # Grounding classification (set by LLM self-tagging)
    grounding_source = Column(String(50), nullable=False, default="unknown")
    # Values: "course_content", "general_knowledge", "uncertain", "mixed", "unknown"

    # Content gap detection
    is_content_gap = Column(Boolean, default=False, index=True)  # True if AI couldn't answer from course
    gap_topic = Column(String(255), nullable=True)  # AI-extracted topic of the gap

    # Metadata
    response_time_ms = Column(Integer, nullable=True)  # How long the response took
    llm_provider = Column(String(50), nullable=True)
    llm_model = Column(String(100), nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    course = relationship("Course", back_populates="chat_logs")


class QuestionCluster(Base):
    """AI-generated clusters of similar questions for FAQ generation."""
    __tablename__ = "question_clusters"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    course_id = Column(String(36), ForeignKey('courses.id'), nullable=False, index=True)

    # Cluster info
    theme = Column(String(255), nullable=False)  # AI-generated theme name
    description = Column(Text, nullable=True)  # AI-generated description
    question_count = Column(Integer, default=0)

    # Representative questions (JSON array of question strings)
    sample_questions = Column(JSON, nullable=True)

    # Suggested FAQ content
    suggested_faq_answer = Column(Text, nullable=True)

    # Analysis metadata
    last_analyzed_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    course = relationship("Course", back_populates="question_clusters")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)

    # Run migrations for existing databases
    _run_migrations()


def _run_migrations():
    """Run database migrations for schema changes."""
    from sqlalchemy import inspect, text

    inspector = inspect(engine)

    # Check if courses table exists
    if 'courses' in inspector.get_table_names():
        columns = [col['name'] for col in inspector.get_columns('courses')]

        # Migration: Add is_active column if it doesn't exist
        if 'is_active' not in columns:
            with engine.connect() as conn:
                conn.execute(text('ALTER TABLE courses ADD COLUMN is_active BOOLEAN DEFAULT 1 NOT NULL'))
                conn.commit()
                print("Migration: Added is_active column to courses table")

    # Check if llm_settings table exists
    if 'llm_settings' in inspector.get_table_names():
        columns = [col['name'] for col in inspector.get_columns('llm_settings')]

        # Migration: Add analytics_logging_level column if it doesn't exist
        if 'analytics_logging_level' not in columns:
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE llm_settings ADD COLUMN analytics_logging_level VARCHAR(20) DEFAULT 'analytics_only' NOT NULL"))
                conn.commit()
                print("Migration: Added analytics_logging_level column to llm_settings table")

    # Check if admin_users table exists for multi-user migration
    if 'admin_users' in inspector.get_table_names():
        columns = [col['name'] for col in inspector.get_columns('admin_users')]

        # Migration: Rename username to email if needed
        if 'username' in columns and 'email' not in columns:
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE admin_users RENAME COLUMN username TO email"))
                conn.commit()
                print("Migration: Renamed username column to email in admin_users table")
            # Refresh columns list after rename
            columns = [col['name'] for col in inspector.get_columns('admin_users')]

        # Migration: Add is_first_user column if it doesn't exist
        if 'is_first_user' not in columns:
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE admin_users ADD COLUMN is_first_user BOOLEAN DEFAULT 0"))
                # Mark existing user as first user and update email
                conn.execute(text("UPDATE admin_users SET is_first_user = 1, email = 'kelly.r.mullaney@gmail.com' WHERE email = 'kellymullaney'"))
                conn.commit()
                print("Migration: Added is_first_user column and migrated existing user")
