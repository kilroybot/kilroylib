from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta

from kilroylib.modules import Module


@dataclass
class TrainingState:
    start_time: datetime
    epochs: int
    updates: int
    module: Module


class StopCondition(ABC):
    @abstractmethod
    def done(self, state: TrainingState) -> bool:
        pass


class NeverStopCondition(StopCondition):
    def done(self, state: TrainingState) -> bool:
        return False


class MaxDurationStopCondition(StopCondition):
    def __init__(self, duration: timedelta) -> None:
        self.duration = duration

    def done(self, state: TrainingState) -> bool:
        return (datetime.utcnow() - state.start_time) >= self.duration


class MaxEpochsStopCondition(StopCondition):
    def __init__(self, epochs: int) -> None:
        self.epochs = epochs

    def done(self, state: TrainingState) -> bool:
        return state.epochs >= self.epochs


class MaxUpdatesStopCondition(StopCondition):
    def __init__(self, updates: int) -> None:
        self.updates = updates

    def done(self, state: TrainingState) -> bool:
        return state.updates >= self.updates
