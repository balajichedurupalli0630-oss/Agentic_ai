"""
models.py

SQLAlchemy ORM Data Models for Document Metadata & Session Tracking.
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, JSON, Text
from db.session import Base


class DocumentMetadata(Base):
    """Document Metadata Table separate from Vector Store."""

    __tablename__ = "document_metadata"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String(255), nullable=False, index=True)
    file_type = Column(String(50), nullable=False)
    total_chunks = Column(Integer, default=0)
    source_path = Column(Text, nullable=True)
    content_type = Column(String(100), default="general")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    extra_metadata = Column(JSON, nullable=True)


class UserSessionRecord(Base):
    """User Session Tracking Table."""

    __tablename__ = "user_sessions"

    session_id = Column(String(255), primary_key=True)
    user_id = Column(String(255), nullable=False, default="anonymous", index=True)
    summary = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
