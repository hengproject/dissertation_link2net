import torch
from os import path, listdir
from PIL import Image as Image
from torch.utils.data import Dataset
from util.dataset_util import is_image_file, load_resized_img
from util.logger_util import get_logger
from util.visualize_util import PIL_to_tensor
from torchvision import transforms
import random


def aug_item_image(a, b, direction,use_augment):
    if use_augment:
        random_angle = random.uniform(-180, 180)
        rotate_func = transforms.RandomRotation(degrees=(random_angle, random_angle))
        a = rotate_func(a)
        b = rotate_func(b)

    if direction == "a2b":
        return a, b
    else:
        return b, a


class DataSetABStyled_Cached(Dataset):
    def __init__(self, img_dir, direction,use_augment):
        super().__init__()
        self.direction = direction
        self.logger = get_logger()
        self.a_dir_path = path.join(img_dir, "a")
        self.b_dir_path = path.join(img_dir, "b")
        self.image_filenames = [x for x in listdir(self.a_dir_path) if is_image_file(x)]
        self.use_augment = use_augment
        transform_list = [
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        self.transform = transforms.Compose(transform_list)
        self.cache_a_image = []
        self.cache_b_image = []
        for index in range(len(self.image_filenames)):
            a = load_resized_img(path.join(self.a_dir_path, self.image_filenames[index]))
            b = load_resized_img(path.join(self.b_dir_path, self.image_filenames[index]))
            a = self.transform(a)
            b = self.transform(b)
            self.cache_a_image.append(a)
            self.cache_b_image.append(b)

    def __getitem__(self, index):
        a = self.cache_a_image[index]
        b = self.cache_b_image[index]
        return aug_item_image(a, b, self.direction,self.use_augment)

    def __len__(self):
        return len(self.image_filenames)


class DataSetABStyled(Dataset):
    def __init__(self, image_dir, direction,use_augment):
        super().__init__()
        self.direction = direction
        self.a_dir_path = path.join(image_dir, "a")
        self.b_dir_path = path.join(image_dir, "b")
        self.image_filenames = [x for x in listdir(self.a_dir_path) if is_image_file(x)]
        self.use_augment = use_augment
        transform_list = [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        a = load_resized_img(path.join(self.a_dir_path, self.image_filenames[index]))
        b = load_resized_img(path.join(self.b_dir_path, self.image_filenames[index]))
        a = self.transform(a)
        b = self.transform(b)
        return aug_item_image(a, b, self.direction,self.use_augment)

    def __len__(self):
        return len(self.image_filenames)
