from sqlalchemy import Column, String, Integer, DateTime, Text, JSON, Boolean, Float, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from database.base import Base

class ClientSession(Base):
    __tablename__ = "client_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), unique=True, index=True, nullable=False)
    
    # Client Information
    client_id = Column(Integer, index=True)
    client_type = Column(String(50))  # managed_brand, one_time, prospect
    client_tier = Column(String(50))  # premium, enterprise, standard
    
    # Session Details
    communication_channel = Column(String(50))  # chat, voice, email
    status = Column(String(50), default="active")  # active, completed, transferred, abandoned
    language = Column(String(10), default="en")
    
    # Interaction Metrics
    message_count = Column(Integer, default=0)
    session_duration = Column(Float, default=0.0)  # in seconds
    sentiment_score = Column(Float)  # -1 to 1 scale
    satisfaction_score = Column(Float)  # 1-5 scale
    
    # AI Learning Context
    client_preferences = Column(JSON)  # Learned preferences
    conversation_style = Column(String(50))  # formal, casual, technical, etc.
    negotiation_style = Column(String(50))  # aggressive, cooperative, price-sensitive
    
    # Session Context
    initial_query = Column(Text)
    resolved_queries = Column(JSON)  # List of resolved topics
    pending_actions = Column(JSON)  # Actions to be completed
    
    # Timestamps
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime)
    last_activity = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    requests = relationship("ClientRequest", back_populates="session")
    messages = relationship("SessionMessage", back_populates="session")
    
    def __repr__(self):
        return f"<ClientSession {self.session_id} ({self.status})>"

class SessionMessage(Base):
    __tablename__ = "session_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("client_sessions.id"), nullable=False)
    
    # Message Content
    message_type = Column(String(50))  # user_message, ai_response, system_notification
    content = Column(Text, nullable=False)
    content_type = Column(String(50), default="text")  # text, voice, image, file
    
    # AI Analysis
    intent = Column(String(100))  # greeting, inquiry, negotiation, complaint, etc.
    entities = Column(JSON)  # Extracted entities (dates, amounts, services, etc.)
    sentiment = Column(Float)  # -1 to 1 scale
    confidence_score = Column(Float)  # AI confidence in understanding
    
    # Response Metrics
    response_time = Column(Float)  # Time to generate response in seconds
    tokens_used = Column(Integer)  # AI tokens consumed
    
    # Metadata
    sequence_number = Column(Integer)  # Order in conversation
    is_escalated = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    session = relationship("ClientSession", back_populates="messages")
    
    def __repr__(self):
        return f"<SessionMessage {self.message_type} ({self.intent})>"