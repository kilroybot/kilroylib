from abc import ABC, abstractmethod
from datetime import datetime, timedelta

from kilroylib.training.offline.state import TrainingState


class StopCondition(ABC):
    @abstractmethod
    def done(self, state: TrainingState) -> bool:
        pass


class NeverStop(StopCondition):
    def done(self, state: TrainingState) -> bool:
        return False


class MaxDuration(StopCondition):
    def __init__(self, duration: timedelta) -> None:
        self.duration = duration

    def done(self, state: TrainingState) -> bool:
        return (datetime.utcnow() - state.start_time) >= self.duration


class MaxEpochs(StopCondition):
    def __init__(self, epochs: int) -> None:
        self.epochs = epochs

    def done(self, state: TrainingState) -> bool:
        return state.epochs >= self.epochs


class MaxUpdates(StopCondition):
    def __init__(self, updates: int) -> None:
        self.updates = updates

    def done(self, state: TrainingState) -> bool:
        return state.updates >= self.updates
