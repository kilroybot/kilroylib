from abc import ABC, abstractmethod
from datetime import datetime, timedelta

from kilroylib.training.state import TrainingState


class StopCondition(ABC):
    @abstractmethod
    async def done(self, state: TrainingState) -> bool:
        pass


class NeverStopCondition(StopCondition):
    async def done(self, state: TrainingState) -> bool:
        return False


class MaxDurationStopCondition(StopCondition):
    def __init__(self, duration: timedelta) -> None:
        self.duration = duration

    async def done(self, state: TrainingState) -> bool:
        return (datetime.utcnow() - state.start_time) >= self.duration


class MaxEpochsStopCondition(StopCondition):
    def __init__(self, epochs: int) -> None:
        self.epochs = epochs

    async def done(self, state: TrainingState) -> bool:
        return state.epochs >= self.epochs


class MaxUpdatesStopCondition(StopCondition):
    def __init__(self, updates: int) -> None:
        self.updates = updates

    async def done(self, state: TrainingState) -> bool:
        return state.updates >= self.updates
