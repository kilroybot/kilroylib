from dataclasses import dataclass
from datetime import datetime

from kilroyshare import OfflineModule


@dataclass
class TrainingState:
    start_time: datetime
    epochs: int
    updates: int
    module: OfflineModule
