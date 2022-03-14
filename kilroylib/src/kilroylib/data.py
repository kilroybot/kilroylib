from abc import ABC, abstractmethod
from collections import Iterable
from pathlib import Path
from types import TracebackType
from typing import (
    AsyncIterable,
    AsyncIterator,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

from aiofiles.tempfile import TemporaryDirectory

from kilroylib.utils import Contextable, run_sync, safe_dump, safe_load

T = TypeVar("T")


class Dataset(ABC, Generic[T], Contextable):
    """Dataset base class.

    Can be used as a context manager.

    Params:
        T (Any): Type of data sample.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Gets number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        pass

    @abstractmethod
    def __getitem__(self, index: Union[int, slice]) -> Union[T, Sequence[T]]:
        """Gets sample at given index.

        Args:
            index (Union[int, slice]): Index of sample in the dataset.
                Int or slice.

        Returns:
            Union[T, Sequence[T]]: Data sample.
                Single if index was int, sequence if index was a slice.
        """
        pass

    @abstractmethod
    async def __agetitem__(
        self, index: Union[int, slice]
    ) -> Union[T, Sequence[T]]:
        """Gets sample at given index asynchronously.

        Args:
            index (Union[int, slice]): Index of sample in the dataset.
                Int or slice.

        Returns:
            Union[T, Sequence[T]]: Data sample.
                Single if index was int, sequence if index was a slice.
        """
        pass


class MemoryCachingDataset(Dataset[T]):
    """Dataset that fetches data and caches it in memory.

    Params:
        T (Any): Type of data sample.
    """

    def __init__(self, iterable: AsyncIterable[T]) -> None:
        """
        Args:
            iterable (AsyncIterable[T]): Async iterable with data samples.
        """
        super().__init__()
        self.iterable = iterable

    async def __aenter__(self) -> "MemoryCachingDataset":
        self.data = [x async for x in self.iterable]
        return self

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: Union[int, slice]) -> Union[T, Sequence[T]]:
        return self.data[index]

    async def __agetitem__(
        self, index: Union[int, slice]
    ) -> Union[T, Sequence[T]]:
        return self.__getitem__(index)


class FileCachingDataset(Dataset[T]):
    """Dataset that fetches data and caches it in filesystem.

    Params:
        T (Any): Type of data sample.
    """

    def __init__(self, iterable: AsyncIterable[T]) -> None:
        """
        Args:
            iterable (AsyncIterable[T]): Async iterable with data samples.
        """
        super().__init__()
        self.iterable = iterable

    def get_path(self, index: int) -> Path:
        return self.tempdir / str(index)

    async def fetch(self) -> int:
        samples = 0
        async for sample in self.iterable:
            await safe_dump(sample, self.get_path(samples))
            samples += 1
        return samples

    async def __aenter__(self) -> "FileCachingDataset":
        self.tempdir_context_manager = TemporaryDirectory()
        self.tempdir = Path(await self.tempdir_context_manager.__aenter__())
        self.n_samples = await self.fetch()
        return self

    async def __aexit__(
        self,
        exctype: Optional[Type[BaseException]],
        excinst: Optional[BaseException],
        exctb: Optional[TracebackType],
    ) -> bool:
        await self.tempdir_context_manager.__aexit__(exctype, excinst, exctb)
        return False  # don't suppress exceptions

    def __len__(self) -> int:
        return self.n_samples

    async def __agetitem__(
        self, index: Union[int, slice]
    ) -> Union[T, Sequence[T]]:
        if isinstance(index, slice):
            return [
                await safe_load(self.get_path(i))
                for i in range(*index.indices(self.__len__()))
            ]

        return await safe_load(self.get_path(index))

    def __getitem__(self, index: Union[int, slice]) -> Union[T, Sequence[T]]:
        return run_sync(self.__agetitem__(index))


class DatasetFactory(ABC, Generic[T]):
    """Utility for dynamic dataset creation.

    Params:
        T (Any): Type of data sample.
    """

    @abstractmethod
    def create(self, data: AsyncIterable[T]) -> Dataset[T]:
        """Creates Dataset from iterator.

        Args:
            data (Iterable[T]): iterable with data samples.

        Returns:
            Dataset[T]: Instance of Dataset created from given iterator.
        """
        pass


class MemoryCachingDatasetFactory(DatasetFactory[T]):
    """DatasetFactory that creates MemoryCachingDataset instances.

    Params:
        T (Any): Type of data sample.
    """

    def create(self, data: AsyncIterable[T]) -> Dataset[T]:
        return MemoryCachingDataset(data)


class FileCachingDatasetFactory(DatasetFactory[T]):
    """DatasetFactory that creates FileCachingDataset instances.

    Params:
        T (Any): Type of data sample.
    """

    def create(self, data: AsyncIterable[T]) -> Dataset[T]:
        return FileCachingDataset(data)


class BatchedDataFetcher(Iterable[List[T]]):
    """Utility for batched data fetching.

    Params:
        T (Any): Type of data sample.
    """

    def __init__(self, dataset: Dataset[T], batch_size: int) -> None:
        """
        Args:
            dataset (Dataset[T]): Dataset to fetch samples from.
            batch_size (int): Number of samples in one batch.
        """
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.len = len(dataset)

    def slices(self) -> Iterator[slice]:
        for i in range(0, self.len, self.batch_size):
            start, stop = i, min(i + self.batch_size, self.len)
            yield slice(start, stop)

    def __iter__(self) -> Iterator[List[T]]:
        """Gets iterator with batches of samples.

        Returns:
            Iterator[List[T]]: Iterator with batches of samples.
                Each batch is a list of samples.
        """
        for index in self.slices():
            yield self.dataset.__getitem__(index)

    async def __aiter__(self) -> AsyncIterator[List[T]]:
        """Gets iterator with batches of samples asynchronously.

        Returns:
            AsyncIterator[List[T]]: Async iterator with batches of samples.
                Each batch is a list of samples.
        """
        for index in self.slices():
            yield await self.dataset.__agetitem__(index)


class DataLoader(Iterable[List[T]]):
    """Utility for iterating over a dataset.

    Params:
        T (Any): Type of data sample.
    """

    DEFAULT_BATCH_SIZE = 1

    def __init__(
        self, dataset: Dataset[T], batch_size: int = DEFAULT_BATCH_SIZE
    ) -> None:
        """
        Args:
            dataset (Dataset[T]): Dataset to iterate over.
            batch_size (int = 1): Number of samples in each batch.
                Defaults to 1.
        """
        super().__init__()
        self.fetcher = BatchedDataFetcher(dataset, batch_size)

    def __iter__(self) -> Iterator[List[T]]:
        """Gets iterator with batches of samples.

        Returns:
            Iterator[List[T]]: Iterator with batches of samples.
                Each batch is a list of samples.
        """
        return iter(self.fetcher)

    def __aiter__(self) -> AsyncIterator[List[T]]:
        """Gets iterator with batches of samples asynchronously.

        Returns:
            AsyncIterator[List[T]]: Async iterator with batches of samples.
                Each batch is a list of samples.
        """
        return self.fetcher.__aiter__()
