from abc import ABC, abstractmethod
from collections import Iterable
from pathlib import Path
from tempfile import TemporaryDirectory
from types import TracebackType
from typing import (
    Callable,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence, Type,
    TypeVar,
    Union,
)

from kilroylib.utils import safe_dump, safe_load

T = TypeVar('T')


class Dataset(ABC, Generic[T]):
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
        return NotImplemented

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
        return NotImplemented

    def __enter__(self) -> 'Dataset':
        return self

    def __exit__(
            self,
            exctype: Optional[Type[BaseException]],
            excinst: Optional[BaseException],
            exctb: Optional[TracebackType]
    ) -> bool:
        return False  # don't suppress exceptions


class MemoryCachingDataset(Dataset[T]):
    """Dataset that fetches data and caches it in memory.

    Params:
        T (Any): Type of data sample.
    """

    def __init__(self, data: Iterator[T]) -> None:
        """
        Args:
            data (Iterator[T]): Iterator with data samples.
        """
        super().__init__()
        self.data = list(data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: Union[int, slice]) -> Union[T, Sequence[T]]:
        return self.data[index]


class FileCachingDataset(Dataset[T]):
    """Dataset that fetches data and caches it in filesystem.

    Params:
        T (Any): Type of data sample.
    """

    def __init__(self, data: Iterator[T]) -> None:
        """
        Args:
            data (Iterator[T]): Iterator with data samples.
        """
        super().__init__()
        self.data = data
        self.tempdir = TemporaryDirectory()
        self.n_samples = self._fetch(self.data, self.tempdir.name)

    @staticmethod
    def _sample_path(
            base_dir: str,
            index: int
    ) -> Path:
        return Path(base_dir) / str(index)

    @staticmethod
    def _fetch(
            data: Iterator[T],
            directory: str
    ) -> int:
        samples = 0
        for sample in data:
            safe_dump(
                sample,
                FileCachingDataset._sample_path(directory, samples)
            )
            samples += 1
        return samples

    def free(self) -> None:
        """Frees the cache. Only for manual use outside context manager."""
        self.tempdir.cleanup()

    def __exit__(
            self,
            exctype: Optional[Type[BaseException]],
            excinst: Optional[BaseException],
            exctb: Optional[TracebackType]
    ) -> bool:
        self.free()
        return False  # don't suppress exceptions

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: Union[int, slice]) -> Union[T, Sequence[T]]:
        if isinstance(index, slice):
            return tuple(
                safe_load(self._sample_path(self.tempdir.name, i))
                for i in range(*index.indices(self.__len__()))
            )

        return safe_load(self._sample_path(self.tempdir.name, index))


class DatasetFactory(ABC, Generic[T]):
    """Utility for dynamic dataset creation.

    Params:
        T (Any): Type of data sample.
    """

    @abstractmethod
    def create(self, iterator_fn: Callable[[], Iterator[T]]) -> Dataset[T]:
        """Creates Dataset from iterator.

        Args:
            iterator_fn (Callable[[], Iterator[T]]): Parameterless function
                that returns an iterator with data samples.

        Returns:
            Dataset[T]: Instance of Dataset created from given iterator.
        """
        return NotImplemented


class MemoryCachingDatasetFactory(DatasetFactory[T]):
    """DatasetFactory that creates MemoryCachingDataset instances.

    Params:
        T (Any): Type of data sample.
    """

    def create(self, iterator_fn: Callable[[], Iterator[T]]) -> Dataset[T]:
        return MemoryCachingDataset(iterator_fn())


class FileCachingDatasetFactory(DatasetFactory[T]):
    """DatasetFactory that creates FileCachingDataset instances.

    Params:
        T (Any): Type of data sample.
    """

    def create(self, iterator_fn: Callable[[], Iterator[T]]) -> Dataset[T]:
        return FileCachingDataset(iterator_fn())


class BatchedDataFetcher(Iterable[List[T]]):
    """Utility for batched data fetching.

    Params:
        T (Any): Type of data sample.
    """

    def __init__(
            self,
            dataset: Dataset[T],
            batch_size: int
    ) -> None:
        """
        Args:
            dataset (Dataset[T]): Dataset to fetch samples from.
            batch_size (int): Number of samples in one batch.
        """
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.len = len(dataset)

    def __iter__(self) -> Iterator[List[T]]:
        """Gets iterator with batches of samples.

        Returns:
            Iterator[List[T]]: Iterator with batches of samples.
                Each batch is a list of samples.
        """
        for i in range(0, self.len, self.batch_size):
            start, stop = i, min(i + self.batch_size, self.len)
            yield self.dataset[start:stop]


class DataLoader(Iterable[List[T]]):
    """Utility for iterating over a dataset.

    Params:
        T (Any): Type of data sample.
    """

    DEFAULT_BATCH_SIZE = 1

    def __init__(
            self,
            dataset: Dataset[T],
            batch_size: int = DEFAULT_BATCH_SIZE
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
