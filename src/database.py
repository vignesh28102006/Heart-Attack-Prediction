import os
from datetime import datetime, timezone
import pandas as pd
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
    desc,
    text,
)
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import declarative_base, sessionmaker

from config import DATABASE_URL, FALLBACK_SQLITE_URL, SQL_ECHO

Base = declarative_base()


class PredictionRecord(Base):
    """SQLAlchemy model representing a patient clinical inference record."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    # Input Clinical Features
    age = Column(Float, nullable=False)
    sex = Column(Integer, nullable=False)
    cp = Column(Integer, nullable=False)
    trestbps = Column(Float, nullable=False)
    chol = Column(Float, nullable=False)
    fbs = Column(Integer, nullable=False)
    restecg = Column(Integer, nullable=False)
    thalach = Column(Float, nullable=False)
    exang = Column(Integer, nullable=False)
    oldpeak = Column(Float, nullable=False)
    slope = Column(Integer, nullable=False)
    ca = Column(Float, nullable=False)
    thal = Column(Float, nullable=False)

    # Model Inferences
    model_used = Column(String(64), nullable=False)
    prediction = Column(Integer, nullable=False)  # 0 or 1
    probability = Column(Float, nullable=False)  # 0.0 to 1.0
    risk_level = Column(String(32), nullable=False)  # Low, Moderate, High

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S")
            if self.created_at
            else "",
            "age": self.age,
            "sex": "Male" if self.sex == 1 else "Female",
            "cp": self.cp,
            "trestbps": self.trestbps,
            "chol": self.chol,
            "fbs": "Yes" if self.fbs == 1 else "No",
            "restecg": self.restecg,
            "thalach": self.thalach,
            "exang": "Yes" if self.exang == 1 else "No",
            "oldpeak": self.oldpeak,
            "slope": self.slope,
            "ca": self.ca,
            "thal": self.thal,
            "model_used": self.model_used,
            "prediction": "Heart Disease" if self.prediction == 1 else "No Disease",
            "probability": f"{self.probability:.1%}",
            "risk_level": self.risk_level,
        }


def get_db_engine():
    """Create database engine targeting PostgreSQL with automatic SQLite fallback."""
    # Attempt connecting to configured PostgreSQL
    try:
        engine = create_engine(
            DATABASE_URL,
            echo=SQL_ECHO,
            connect_args={"connect_timeout": 3}
            if "postgresql" in DATABASE_URL
            else {},
        )
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print(f"Connected to primary database: {engine.url.render_as_string(hide_password=True)}")
        return engine
    except Exception as e:
        print(f"PostgreSQL connection failed ({e}). Falling back to local SQLite: {FALLBACK_SQLITE_URL}")
        fallback_engine = create_engine(
            FALLBACK_SQLITE_URL,
            echo=SQL_ECHO,
            connect_args={"check_same_thread": False},
        )
        return fallback_engine


_engine = None
_SessionFactory = None


def init_db(engine=None):
    """Initialize database tables."""
    global _engine, _SessionFactory
    if engine is not None:
        _engine = engine
    elif _engine is None:
        _engine = get_db_engine()

    Base.metadata.create_all(_engine)
    _SessionFactory = sessionmaker(bind=_engine)
    return _engine


def get_session():
    """Get a SQLAlchemy session instance."""
    global _SessionFactory
    if _SessionFactory is None:
        init_db()
    return _SessionFactory()


def determine_risk_level(prob: float) -> str:
    """Classify probability into standard clinical risk categories."""
    if prob < 0.35:
        return "Low Risk"
    elif prob <= 0.65:
        return "Moderate Risk"
    else:
        return "High Risk"


def save_prediction(
    feature_dict: dict,
    model_name: str,
    prediction: int,
    probability: float,
    session=None,
) -> PredictionRecord:
    """Save an inference record into the database."""
    close_session = False
    if session is None:
        session = get_session()
        close_session = True

    try:
        record = PredictionRecord(
            age=float(feature_dict["age"]),
            sex=int(feature_dict["sex"]),
            cp=int(feature_dict["cp"]),
            trestbps=float(feature_dict["trestbps"]),
            chol=float(feature_dict["chol"]),
            fbs=int(feature_dict["fbs"]),
            restecg=int(feature_dict["restecg"]),
            thalach=float(feature_dict["thalach"]),
            exang=int(feature_dict["exang"]),
            oldpeak=float(feature_dict["oldpeak"]),
            slope=int(feature_dict["slope"]),
            ca=float(feature_dict["ca"]),
            thal=float(feature_dict["thal"]),
            model_used=model_name,
            prediction=int(prediction),
            probability=float(probability),
            risk_level=determine_risk_level(probability),
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return record
    except Exception:
        session.rollback()
        raise
    finally:
        if close_session:
            session.close()


def get_prediction_history(limit: int = 100, session=None) -> list[dict]:
    """Retrieve history of predictions ordered by recency."""
    close_session = False
    if session is None:
        session = get_session()
        close_session = True

    try:
        records = (
            session.query(PredictionRecord)
            .order_by(desc(PredictionRecord.created_at))
            .limit(limit)
            .all()
        )
        return [r.to_dict() for r in records]
    finally:
        if close_session:
            session.close()


def clear_history(session=None):
    """Clear all records from predictions table."""
    close_session = False
    if session is None:
        session = get_session()
        close_session = True

    try:
        session.query(PredictionRecord).delete()
        session.commit()
    finally:
        if close_session:
            session.close()


if __name__ == "__main__":
    init_db()
    # Test insertion
    sample = {
        "age": 58,
        "sex": 1,
        "cp": 2,
        "trestbps": 140,
        "chol": 240,
        "fbs": 0,
        "restecg": 1,
        "thalach": 160,
        "exang": 0,
        "oldpeak": 1.2,
        "slope": 1,
        "ca": 0,
        "thal": 2,
    }
    rec = save_prediction(sample, "Baseline Test", 1, 0.78)
    print(f"Record saved with ID: {rec.id}, Risk: {rec.risk_level}")
    history = get_prediction_history(5)
    print(f"Retrieved {len(history)} historical records.")
