from abc import ABC, abstractmethod
from datetime import datetime, timedelta

from kilroylib.training.online.state import TrainingState


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


class MaxEpisodes(StopCondition):
    def __init__(self, episodes: int) -> None:
        self.episodes = episodes

    def done(self, state: TrainingState) -> bool:
        return state.episodes >= self.episodes


class MaxUpdates(StopCondition):
    def __init__(self, updates: int) -> None:
        self.updates = updates

    def done(self, state: TrainingState) -> bool:
        return state.updates >= self.updates
