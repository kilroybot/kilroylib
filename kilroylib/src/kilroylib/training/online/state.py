from dataclasses import dataclass
from datetime import datetime

from kilroyshare import OnlineModule


@dataclass
class TrainingState:
    start_time: datetime
    updates: int
    episodes: int
    module: OnlineModule
