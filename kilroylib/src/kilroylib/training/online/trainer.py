from asyncio import sleep
from datetime import datetime, timedelta
from typing import Dict, Generic, Hashable, List, Sequence, Tuple, TypeVar

from kilroyshare import Face, OnlineModule

from kilroylib.training.online.state import TrainingState
from kilroylib.training.online.stop import (
    MaxEpisodes,
    StopCondition,
)
from kilroylib.utils import abackground, background

KI = TypeVar("KI", bound=Hashable)
KE = TypeVar("KE", bound=Hashable)
V = TypeVar("V")


class PostGenerator(Generic[KI, V]):
    def __init__(self, n: int) -> None:
        self.n = n

    async def generate(self, module: OnlineModule) -> Dict[KI, V]:
        return {
            internal_id: post
            for internal_id, post in await background(module.sample, self.n)
        }


class PostScheduler(Generic[KE, V]):
    def __init__(self, cooldown: timedelta) -> None:
        self.cooldown = cooldown

    async def post(self, posts: Sequence[V], face: Face) -> List[KE]:
        post_ids = []
        for post in posts:
            post_id = await abackground(face.post(post))
            post_ids.append(post_id)
            await self.wait()
        return post_ids

    async def wait(self) -> None:
        await sleep(self.cooldown.seconds)


class Trainer(Generic[KI, KE, V]):
    DEFAULT_STOP_CONDITION = MaxEpisodes(1)
    DEFAULT_POST_GENERATOR = PostGenerator(1)
    DEFAULT_POST_SCHEDULER = PostScheduler(timedelta(seconds=1))
    DEFAULT_EPISODE_ITERATIONS = 1
    DEFAULT_EPISODES_PER_UPDATE = 1

    def __init__(
        self,
        stop_condition: StopCondition = DEFAULT_STOP_CONDITION,
        generator: PostGenerator = DEFAULT_POST_GENERATOR,
        scheduler: PostScheduler = DEFAULT_POST_SCHEDULER,
        episode_iterations: int = DEFAULT_EPISODE_ITERATIONS,
        episodes_per_update: int = DEFAULT_EPISODES_PER_UPDATE,
    ) -> None:
        super().__init__()
        self.stop_condition = stop_condition
        self.generator = generator
        self.scheduler = scheduler
        self.episode_iterations = episode_iterations
        self.episodes_per_update = episodes_per_update

    @staticmethod
    def get_starting_state(module: OnlineModule) -> TrainingState:
        return TrainingState(
            start_time=datetime.utcnow(),
            updates=0,
            episodes=0,
            module=module,
        )

    def should_stop(self, state: TrainingState) -> bool:
        return self.stop_condition.done(state)

    async def generate(self, module: OnlineModule) -> Tuple[List[KI], List[V]]:
        generated = await self.generator.generate(module)
        keys = list(generated.keys())
        posts = [generated[key] for key in keys]
        return keys, posts

    async def post(self, posts: List[V], face: Face) -> List[KE]:
        return await self.scheduler.post(posts, face)

    @staticmethod
    async def score(post_ids: Sequence[KE], face) -> List[float]:
        return [await abackground(face.score(post_id)) for post_id in post_ids]

    @staticmethod
    def map_scores(
        internal_post_ids: Sequence[KE], scores: Sequence[float]
    ) -> Dict[KE, float]:
        return {
            post_id: score for post_id, score in zip(internal_post_ids, scores)
        }

    async def fit(
        self, module: OnlineModule, ids: Sequence[KI], scores: Sequence[float]
    ) -> None:
        scores = self.map_scores(ids, scores)
        for _ in range(self.episode_iterations):
            metrics = await background(module.fit, scores)

    def should_update(self, episode: int) -> bool:
        return (episode + 1) % self.episodes_per_update == 0

    @staticmethod
    async def step(module: OnlineModule) -> OnlineModule:
        return await background(module.step)

    async def train(self, module: OnlineModule, face: Face) -> OnlineModule:
        state = self.get_starting_state(module)
        while not self.should_stop(state):
            internal_ids, posts = await self.generate(state.module)
            external_ids = await self.post(posts, face)
            scores = await self.score(external_ids, face)
            await self.fit(state.module, internal_ids, scores)
            if self.should_update(state.episodes):
                state.module = await self.step(state.module)
                state.updates += 1
                if self.should_stop(state):
                    return state.module
            state.episodes += 1
        return module
