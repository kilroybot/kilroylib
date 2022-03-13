import time
from datetime import datetime, timedelta
from typing import Dict, Generic, Hashable, List, Sequence, Tuple, TypeVar

from kilroyshare import Face

from kilroylib.modules import Module
from kilroylib.training.stop import (
    MaxUpdatesStopCondition,
    StopCondition,
    TrainingState,
)

KI = TypeVar("KI", bound=Hashable)
KE = TypeVar("KE", bound=Hashable)
V = TypeVar("V")


class PostGenerator(Generic[KI, V]):
    def __init__(self, n: int) -> None:
        self.n = n

    def generate(self, module: Module) -> Dict[KI, V]:
        posts = (module.generate() for _ in range(self.n))
        return {internal_id: post for internal_id, post in posts}


class PostScheduler(Generic[KE, V]):
    def __init__(self, cooldown: timedelta) -> None:
        self.cooldown = cooldown

    def post(self, posts: Sequence[V], face: Face) -> List[KE]:
        post_ids = []
        for post in posts:
            post_id = face.post(post)
            post_ids.append(post_id)
            self.wait()
        return post_ids

    def wait(self) -> None:
        time.sleep(self.cooldown.seconds)


class PostScorer(Generic[KE]):
    def score(self, post_id: KE, face: Face) -> float:
        return face.score(post_id)


class OnlineTrainer(Generic[KI, KE, V]):
    DEFAULT_STOP_CONDITION = MaxUpdatesStopCondition(1)
    DEFAULT_POST_GENERATOR = PostGenerator(1)
    DEFAULT_POST_SCHEDULER = PostScheduler(timedelta(seconds=1))
    DEFAULT_POST_SCORER = PostScorer()

    def __init__(
        self,
        stop_condition: StopCondition = DEFAULT_STOP_CONDITION,
        generator: PostGenerator = DEFAULT_POST_GENERATOR,
        scheduler: PostScheduler = DEFAULT_POST_SCHEDULER,
        scorer: PostScorer = DEFAULT_POST_SCORER,
    ) -> None:
        super().__init__()
        self.stop_condition = stop_condition
        self.generator = generator
        self.scheduler = scheduler
        self.scorer = scorer

    def train(self, module: Module, face: Face) -> Module:
        state = TrainingState(
            start_time=datetime.utcnow(),
            epochs=0,
            updates=0,
            module=module,
        )
        while not self.stop_condition.done(state):
            internal_ids, posts = self.generate(module)
            external_ids = self.scheduler.post(posts, face)
            scores = self.score(external_ids, face)
            module = module.reinforce(self.map_scores(internal_ids, scores))
            state.updates += 1
            state.module = module
        return module

    def generate(self, module) -> Tuple[List[KI], List[V]]:
        generated = self.generator.generate(module)
        keys = list(generated.keys())
        posts = [generated[key] for key in keys]
        return keys, posts

    def score(self, post_ids: Sequence[KE], face) -> List[float]:
        return [self.scorer.score(post_id, face) for post_id in post_ids]

    @staticmethod
    def map_scores(
        internal_post_ids: Sequence[KE], scores: Sequence[float]
    ) -> Dict[KE, float]:
        return {
            post_id: score for post_id, score in zip(internal_post_ids, scores)
        }
