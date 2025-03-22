from pathlib import Path
from PIL import Image
import numpy as np
import random

from model.classification import ClassificationType, DatasetType


class DataManager:
    @staticmethod
    def count_number_of_images(class_type: ClassificationType):
        CLASS_DATA_PATH = Path(__file__).parent.parent / "data" / "processed"
        TRAIN_PATH = CLASS_DATA_PATH / "train" / class_type.value
        TEST_PATH = CLASS_DATA_PATH / "test" / class_type.value
        VAL_PATH = CLASS_DATA_PATH / "val" / class_type.value

        all_dirs = [TRAIN_PATH, TEST_PATH, VAL_PATH]
        number_of_images = sum([1 for dir in all_dirs for file in dir.iterdir()])

        return number_of_images

    @staticmethod
    def get_image(
        class_type: ClassificationType = None,
        dataset_type: DatasetType = None,
        image_name: str = None,
    ) -> np.ndarray:
        if not class_type:
            class_type = ClassificationType.get_all_types()[
                random.randint(0, ClassificationType.get_number_of_types() - 1)
            ]
        if not dataset_type:
            dataset_type = DatasetType.get_all_types()[
                random.randint(0, DatasetType.get_number_of_types() - 1)
            ]

        IMAGES_PATH = (
            Path(__file__).parent.parent
            / "data"
            / "processed"
            / dataset_type.value
            / class_type.value
        )
        if not image_name:
            number_of_images = sum([1 for file in IMAGES_PATH.iterdir()])

            image_idx = random.randint(0, number_of_images - 1)
            image_path = None
            for i, file in enumerate(IMAGES_PATH.iterdir()):
                if i == image_idx:
                    image_path = file
                    break
        else:
            image_path = None
            for file in IMAGES_PATH.iterdir():
                if str(file).endswith(image_name):
                    image_path = file
                    break

        image = Image.open(image_path)
        image_array = np.array(image)
        return image_array
