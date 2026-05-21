from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone

db = SQLAlchemy()

class OCRResult(db.Model):
    __tablename__ = 'ocr_results'

    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    extracted_text = db.Column(db.Text, nullable=True)
    confidence = db.Column(db.Float, nullable=True)
    engine_used = db.Column(db.String(50), nullable=True)
    pipeline = db.Column(db.String(50), nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def __init__(self, filename, extracted_text=None, confidence=None, engine_used=None, pipeline=None, **kwargs):
        super().__init__(**kwargs)
        self.filename = filename
        self.extracted_text = extracted_text
        self.confidence = confidence
        self.engine_used = engine_used
        self.pipeline = pipeline

    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "extracted_text": self.extracted_text,
            "confidence": self.confidence,
            "engine_used": self.engine_used,
            "pipeline": self.pipeline,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
