import os.path
from .base_dataset import DataSetABStyled, DataSetABStyled_Cached


def get_train_dataset(root_dir, dataset_name, direction, ram_cached=False, use_augment=True):
    train_dir_path = os.path.join(root_dir, dataset_name, "train")
    if ram_cached:
        return DataSetABStyled_Cached(train_dir_path, direction,use_augment)
    else:
        return DataSetABStyled(train_dir_path, direction,use_augment)


def get_test_dataset(root_dir, dataset_name, direction, ram_cached=False, use_augment=False):
    test_dir_path = os.path.join(root_dir, dataset_name, "test")
    if ram_cached:
        return DataSetABStyled_Cached(test_dir_path, direction,use_augment)
    else:
        return DataSetABStyled(test_dir_path, direction,use_augment)
