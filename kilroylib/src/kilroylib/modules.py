from abc import ABC, abstractmethod
from typing import (
    Collection,
    Dict,
    Generic,
    Hashable,
    Iterator,
    Tuple,
    TypeVar,
)

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class Module(ABC, Generic[K, V]):
    """Base class for modules.

    Params:
        K (Hashable): Type of internal identifier of generated samples.
        V: Type of sample.
    """

    @abstractmethod
    def generate(self, n: int = 1) -> Iterator[Tuple[K, V]]:
        """Generates new sample.

        Args:
            n (int, default: 1): How many samples to generate.

        Returns:
            Iterator[Tuple[K, V]]: Iterator of tuples with internal identifier
                of generated sample and generated sample itself.
        """
        pass

    @abstractmethod
    def mimic(self, samples: Collection[V]) -> "Module":
        """Learns to mimic existing samples.

        Args:
            samples (Collection[V]): Collection of existing samples.

        Returns:
            KilroyModule: Instance of KilroyModule after learning (can be self).
        """
        pass

    @abstractmethod
    def reinforce(self, scores: Dict[K, float]) -> "Module":
        """Learns from previous samples' scores.

        Args:
            scores (Dict[K, float]): Dictionary with internal samples
                identifiers as keys and samples scores as values.

        Returns:
            KilroyModule: Instance of KilroyModule after learning (can be self).
        """
        pass
