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

from kilroyshare import Face

from kilroylib.data import (
    DataLoader,
    Dataset,
    DatasetFactory,
    MemoryCachingDatasetFactory,
)
from kilroylib.modules import Module
from kilroylib.training.state import TrainingState
from kilroylib.training.stop import (
    MaxEpochsStopCondition,
    StopCondition,
)

V = TypeVar("V")


class PostsLoader(Generic[V]):
    def __init__(
        self, batch_size: int, dataset_factory: DatasetFactory[Tuple[Any, V]]
    ) -> None:
        self.batch_size = batch_size
        self.dataset_factory = dataset_factory

    async def load(self, face: Face) -> AsyncIterator[List[V]]:
        async with self.create_dataset(face) as dataset:
            async for batch in self.iterate_dataset(dataset):
                yield self.map_samples(batch)

    def create_dataset(self, face: Face) -> Dataset[Tuple[Any, V]]:
        return self.dataset_factory.create(face.scrap())

    def iterate_dataset(
        self, dataset: Dataset[Tuple[Any, V]]
    ) -> DataLoader[Tuple[Any, V]]:
        return DataLoader(dataset, batch_size=self.batch_size)

    @staticmethod
    def map_samples(samples: Iterable[Tuple[Any, V]]) -> List[V]:
        return [sample[1] for sample in samples]


class OfflineTrainer(Generic[V]):
    DEFAULT_STOP_CONDITION = MaxEpochsStopCondition(epochs=1)
    DEFAULT_POSTS_LOADER = PostsLoader(
        batch_size=1, dataset_factory=MemoryCachingDatasetFactory()
    )

    def __init__(
        self,
        stop_condition: StopCondition = DEFAULT_STOP_CONDITION,
        posts_loader: PostsLoader[V] = DEFAULT_POSTS_LOADER,
    ) -> None:
        super().__init__()
        self.stop_condition = stop_condition
        self.posts_loader = posts_loader

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
        while not await self.stop_condition.done(state):  # epoch loop
            async for posts in self.posts_loader.load(face):  # batch loop
                state.module = await state.module.mimic(posts)
                state.updates += 1  # update per batch
                if await self.stop_condition.done(state):
                    return state.module
            state.epochs += 1
        return state.module
