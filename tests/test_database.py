import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.database import Base, PredictionRecord, save_prediction, get_prediction_history, clear_history


@pytest.fixture
def db_session():
    """Create in-memory SQLite engine for testing isolated from production DB."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_save_prediction_record(db_session):
    """Verify inserting and querying a prediction record."""
    sample = {
        "age": 55,
        "sex": 0,
        "cp": 1,
        "trestbps": 120,
        "chol": 210,
        "fbs": 0,
        "restecg": 0,
        "thalach": 165,
        "exang": 0,
        "oldpeak": 0.5,
        "slope": 2,
        "ca": 0,
        "thal": 2,
    }

    record = save_prediction(
        feature_dict=sample,
        model_name="XGBoost Test",
        prediction=0,
        probability=0.15,
        session=db_session,
    )

    assert record.id is not None
    assert record.prediction == 0
    assert record.risk_level == "Low Risk"
    assert record.model_used == "XGBoost Test"


def test_get_prediction_history(db_session):
    """Verify fetching history list and clearing."""
    sample = {
        "age": 62,
        "sex": 1,
        "cp": 0,
        "trestbps": 140,
        "chol": 260,
        "fbs": 1,
        "restecg": 1,
        "thalach": 120,
        "exang": 1,
        "oldpeak": 2.5,
        "slope": 1,
        "ca": 2,
        "thal": 3,
    }

    save_prediction(sample, "RF Test", 1, 0.88, session=db_session)
    history = get_prediction_history(limit=10, session=db_session)
    assert len(history) == 1
    assert history[0]["risk_level"] == "High Risk"
    assert history[0]["prediction"] == "Heart Disease"

    clear_history(session=db_session)
    cleared = get_prediction_history(limit=10, session=db_session)
    assert len(cleared) == 0
