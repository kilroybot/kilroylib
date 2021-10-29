from abc import ABC, abstractmethod
from typing import Collection, Dict, Generic, Hashable, Tuple, TypeVar

from kilroyshare import PostData

T = TypeVar('T', bound=Hashable)


class KilroyModule(ABC, Generic[T]):
    """KilroyModule base class.

    Params:
        T (Hashable): Type of internal id of generated posts.
    """

    @abstractmethod
    def generate(self) -> Tuple[T, PostData]:
        """Generates new post.

        Returns:
            Tuple[T, PostData]: Internal id of generated post
                and generated post data.
        """
        return NotImplemented

    @abstractmethod
    def mimic(self, posts: Collection[PostData]) -> 'KilroyModule':
        """Learns to mimic existing posts.

        Args:
            posts (Collection[PostData]): Collection of existing posts.

        Returns:
            KilroyModule: Instance of KilroyModule after learning (can be self).
        """
        return NotImplemented

    @abstractmethod
    def reinforce(self, scores: Dict[T, float]) -> 'KilroyModule':
        """Learns from previous posts' scores.

        Args:
            scores (Dict[T, float]): Dictionary with internal post ids as keys
                and post scores as values.

        Returns:
            KilroyModule: Instance of KilroyModule after learning (can be self).
        """
        return NotImplemented
