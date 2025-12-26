"""CRUD operations for database models."""

from app.crud.job import job_crud
from app.crud.ml_model import model_crud
from app.crud.prediction import prediction_crud

__all__ = ["model_crud", "prediction_crud", "job_crud"]
