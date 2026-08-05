"""What kind of problem a dataset represents."""

from __future__ import annotations

from enum import Enum

__all__ = ["Task"]


class Task(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"

    def __str__(self) -> str:
        return self.value
