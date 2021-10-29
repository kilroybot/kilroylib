import time
from abc import ABC, abstractmethod

from kilroylib.data import (
    DataLoader,
    Dataset,
    DatasetFactory,
    FileCachingDatasetFactory,
)
from kilroylib.modules import KilroyModule
from kilroyshare import KilroyFace, PostData


class FineTuner(ABC):
    @abstractmethod
    def tune(
            self,
            module: KilroyModule,
            dataset: Dataset[PostData]
    ) -> KilroyModule:
        return NotImplemented


class BasicFineTuner(FineTuner):
    DEFAULT_MAX_EPOCHS = 100
    DEFAULT_BATCH_SIZE = 1

    def __init__(
            self,
            epochs: int = DEFAULT_MAX_EPOCHS,
            batch_size: int = DEFAULT_BATCH_SIZE
    ) -> None:
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size

    def tune(
            self,
            module: KilroyModule,
            dataset: Dataset[PostData]
    ) -> KilroyModule:
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size
        )

        for _ in range(self.epochs):
            for batch in dataloader:
                posts = [sample[1] for sample in batch]
                module = module.mimic(posts)

        return module


class KilroyTrainer:
    DEFAULT_DATASET_FACTORY = FileCachingDatasetFactory()
    DEFAULT_TUNER = BasicFineTuner()

    def __init__(
            self,
            face: KilroyFace,
            module: KilroyModule,
            dataset_factory: DatasetFactory = DEFAULT_DATASET_FACTORY,
            tuner: FineTuner = DEFAULT_TUNER
    ) -> None:
        super().__init__()
        self.face = face
        self.module = module
        self.dataset_factory = dataset_factory
        self.tuner = tuner

    def fine_tune(self) -> 'KilroyTrainer':
        with self.dataset_factory.create(self.face.scrap) as dataset:
            self.module = self.tuner.tune(self.module, dataset)

        return self

    def run(
            self,
            steps: int = 10,
            post_rate: int = 1,
            update_rate: int = 10
    ) -> 'KilroyTrainer':
        for _ in range(steps):
            post_ids = []
            for _ in range(update_rate):
                module_post_id, post = self.module.generate()
                face_post_id = self.face.post(post)
                post_ids.append((module_post_id, face_post_id))
                time.sleep(post_rate)
            self.module = self.module.reinforce({
                module_post_id: self.face.score(face_post_id)
                for module_post_id, face_post_id in post_ids
            })

        return self
