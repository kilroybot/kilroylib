from abc import ABC, abstractmethod
from typing import Collection, Dict, Generic, Hashable, Tuple, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class Module(ABC, Generic[K, V]):
    """Base class for modules.

    Params:
        K (Hashable): Type of internal identifier of generated samples.
        V: Type of sample.
    """

    @abstractmethod
    def generate(self) -> Tuple[K, V]:
        """Generates new sample.

        Returns:
            Tuple[K, V]: Internal identifier of generated sample and generated
                sample itself.
        """
        return NotImplemented

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
