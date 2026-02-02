import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex, H5File


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example



class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)


class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)


class CustomVideoTrain(CustomBase):
    def __init__(self, size, training_h5file_path, split_file_path=None, formatter="{}"):
        super().__init__()
        self.data = H5File(
            h5_file_path=training_h5file_path, 
            split_file_path=split_file_path, 
            size=size, 
            random_crop=False, 
            split="train", 
            formatter=formatter
        )


class CustomVideoTest(CustomBase):
    def __init__(self, size, test_h5file_path, split_file_path=None, formatter="{}"):
        super().__init__()
        self.data = H5File(
            h5_file_path=test_h5file_path, 
            split_file_path=split_file_path, 
            size=size, 
            random_crop=False, 
            split="test",
            formatter=formatter
        )
