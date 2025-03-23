from enum import Enum


class ClassificationType(Enum):
    BABIES = "babies"
    MEN = "men"
    WOMEN = "women"
    NONE = "none"

    @classmethod
    def get_all_types(cls):
        return [class_type for class_type in cls]

    @classmethod
    def get_names_of_all_types(cls):
        return [class_type.value for class_type in cls]

    @classmethod
    def get_number_of_types(cls):
        return len(ClassificationType.get_all_types())

    @classmethod
    def get_class_to_idx(cls):
        return {type_name.value: i for i, type_name in enumerate(cls.get_all_types())}


class DatasetType(Enum):
    TEST = "test"
    TRAIN = "train"
    VAL = "val"

    @classmethod
    def get_all_types(cls):
        return [class_type for class_type in cls]

    @classmethod
    def get_number_of_types(cls):
        return len(DatasetType.get_all_types())
