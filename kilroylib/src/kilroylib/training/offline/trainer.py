from asyncio import Lock
from datetime import datetime
from typing import (
    Any,
    AsyncIterator,
    Generic,
    Iterable,
    List,
    Tuple,
    TypeVar,
)

from kilroyshare import Face, OfflineModule

from kilroylib.data import (
    DataLoader,
    Dataset,
    DatasetFactory,
    MemoryCachingDatasetFactory,
)
from kilroylib.training.offline.state import TrainingState
from kilroylib.training.offline.stop import (
    MaxEpochs,
    StopCondition,
)
from kilroylib.utils import aenumerate, background

V = TypeVar("V")


class PostsLoader(Generic[V]):
    def __init__(
        self, batch_size: int, dataset_factory: DatasetFactory[Tuple[Any, V]]
    ) -> None:
        self.batch_size = batch_size
        self.dataset_factory = dataset_factory

    async def load(self, face: Face) -> Dataset[Tuple[Any, V]]:
        it = await background(face.scrap)
        return self.dataset_factory.create(it)

    async def iterate(
        self, dataset: Dataset[Tuple[Any, V]]
    ) -> AsyncIterator[List[V]]:
        async for batch in DataLoader(dataset, batch_size=self.batch_size):
            yield self._map_samples(batch)

    @staticmethod
    def _map_samples(samples: Iterable[Tuple[Any, V]]) -> List[V]:
        return [sample[1] for sample in samples]


class Trainer(Generic[V]):
    DEFAULT_STOP_CONDITION = MaxEpochs(epochs=1)
    DEFAULT_POSTS_LOADER = PostsLoader(
        batch_size=1, dataset_factory=MemoryCachingDatasetFactory()
    )
    DEFAULT_BATCH_ITERATIONS = 1
    DEFAULT_BATCHES_PER_UPDATE = 1

    def __init__(
        self,
        stop_condition: StopCondition = DEFAULT_STOP_CONDITION,
        posts_loader: PostsLoader[V] = DEFAULT_POSTS_LOADER,
        batch_iterations: int = DEFAULT_BATCH_ITERATIONS,
        batches_per_update: int = DEFAULT_BATCHES_PER_UPDATE,
    ) -> None:
        super().__init__()
        self.stop_condition = stop_condition
        self.posts_loader = posts_loader
        self.batch_iterations = batch_iterations
        self.batches_per_update = batches_per_update
        self._lock = None
        self._stop_requested = False

    @staticmethod
    def _get_starting_state(module: OfflineModule) -> TrainingState:
        return TrainingState(
            start_time=datetime.utcnow(),
            epochs=0,
            updates=0,
            module=module,
        )

    async def _load(self, face: Face) -> Dataset[Tuple[Any, V]]:
        return await self.posts_loader.load(face)

    async def _should_stop(self, state: TrainingState) -> bool:
        async with self._lock:
            return self._stop_requested or self.stop_condition.done(state)

    def _iterate(
        self, dataset: Dataset[Tuple[Any, V]]
    ) -> AsyncIterator[List[V]]:
        return self.posts_loader.iterate(dataset)

    async def _fit(self, module: OfflineModule, batch: List[V]) -> None:
        for _ in range(self.batch_iterations):
            metrics = await background(module.fit, batch)

    def _should_update(self, batch_index: int) -> bool:
        return (batch_index + 1) % self.batches_per_update == 0

    @staticmethod
    async def _step(module: OfflineModule) -> OfflineModule:
        return await background(module.step)

    async def init(self) -> None:
        self._lock = Lock()

    async def _on_ended(self) -> None:
        async with self._lock:
            self._stop_requested = False

    async def start(self, module: OfflineModule, face: Face) -> OfflineModule:
        state = self._get_starting_state(module)
        async with (await self._load(face)) as dataset:
            # epoch loop
            while not await self._should_stop(state):
                # batch loop
                async for i, batch in aenumerate(self._iterate(dataset)):
                    await self._fit(state.module, batch)
                    if self._should_update(i):
                        state.module = await self._step(state.module)
                        state.updates += 1
                        if await self._should_stop(state):
                            break
                state.epochs += 1
            await self._on_ended()
            return state.module

    async def stop(self) -> None:
        async with self._lock:
            self._stop_requested = True
