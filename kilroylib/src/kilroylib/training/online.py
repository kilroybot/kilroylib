import asyncio
from datetime import datetime, timedelta
from typing import Dict, Generic, Hashable, List, Sequence, Tuple, TypeVar

from kilroyshare import Face

from kilroylib.modules import Module
from kilroylib.training.state import TrainingState
from kilroylib.training.stop import (
    MaxUpdatesStopCondition,
    StopCondition,
)

KI = TypeVar("KI", bound=Hashable)
KE = TypeVar("KE", bound=Hashable)
V = TypeVar("V")


class PostGenerator(Generic[KI, V]):
    def __init__(self, n: int) -> None:
        self.n = n

    async def generate(self, module: Module) -> Dict[KI, V]:
        return {
            internal_id: post
            async for internal_id, post in module.generate(self.n)
        }


class PostScheduler(Generic[KE, V]):
    def __init__(self, cooldown: timedelta) -> None:
        self.cooldown = cooldown

    async def post(self, posts: Sequence[V], face: Face) -> List[KE]:
        post_ids = []
        for post in posts:
            post_id = await face.post(post)
            post_ids.append(post_id)
            await self.wait()
        return post_ids

    async def wait(self) -> None:
        await asyncio.sleep(self.cooldown.seconds)


class PostScorer(Generic[KE]):
    async def score(self, post_id: KE, face: Face) -> float:
        return await face.score(post_id)


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

    @staticmethod
    def get_starting_state(module: Module) -> TrainingState:
        return TrainingState(
            start_time=datetime.utcnow(),
            epochs=0,
            updates=0,
            module=module,
        )

    async def train(self, module: Module, face: Face) -> Module:
        state = self.get_starting_state(module)
        while not await self.stop_condition.done(state):  # update loop
            internal_ids, posts = await self.generate(state.module)
            external_ids = await self.scheduler.post(posts, face)
            scores = await self.score(external_ids, face)
            state.module = await state.module.reinforce(
                self.map_scores(internal_ids, scores)
            )
            state.updates += 1
        return module

    async def generate(self, module) -> Tuple[List[KI], List[V]]:
        generated = await self.generator.generate(module)
        keys = list(generated.keys())
        posts = [generated[key] for key in keys]
        return keys, posts

    async def score(self, post_ids: Sequence[KE], face) -> List[float]:
        return [await self.scorer.score(post_id, face) for post_id in post_ids]

    @staticmethod
    def map_scores(
        internal_post_ids: Sequence[KE], scores: Sequence[float]
    ) -> Dict[KE, float]:
        return {
            post_id: score for post_id, score in zip(internal_post_ids, scores)
        }
