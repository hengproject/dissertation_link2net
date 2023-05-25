import random
import time
from os import path, listdir
import torch
from torch.utils.data import DataLoader

from util import visualize_util
from data.ect_dataset import get_test_dataset, get_train_dataset
from util.dataset_util import is_image_file, load_resized_img, PIL_to_tensor
from data.base_dataset import DataSetABStyled
from util.save_util import save_pil_image
from util.visualize_util import tensor_to_PIL
from PIL import Image as Image
from torchvision import transforms


def test1():
    test_tensor_base = get_test_dataset(r'F:\datasets\pix2pix\\', 'ect', 'a2b', ram_cached=False)
    for i in range(3):
        T1 = time.perf_counter()
        print(i)
        test_tensor = test_tensor_base[0]
        T2 = time.perf_counter()
        print(f"取从无缓存取一个时间为{(T2 - T1) * 1000}毫秒")
        print(test_tensor[0].shape)
        tensor_concat = torch.concat([test_tensor[0], test_tensor[1]], dim=1)
        print(tensor_concat.size())
        pil = visualize_util.tensor_to_PIL(tensor_concat)
        pil.show()


def test2():
    T1 = time.perf_counter()
    random.random()
    w_offset = random.randint(0, max(0, 286 - 256 - 1))
    h_offset = random.randint(0, max(0, 286 - 256 - 1))
    T2 = time.perf_counter()
    print(f"随机数{(T2 - T1) * 1000}毫秒")


def test3():
    tensor = torch.tensor([1, 2, 3])
    to_tensor = transforms.ToTensor()
    t = to_tensor(tensor)
    print(t)  # ERROR


def NotRGBImgtest():
    image_open1 = Image.open(r'F:\datasets\pix2pix\ect\train\a\4.png')
    image_open2 = Image.open(r'F:\datasets\pix2pix\ect\train\a\4.png').convert('RGB')
    print(image_open1.size)
    print(image_open2.size)


def test4():
    test_dataset = get_test_dataset(r'F:\datasets\pix2pix\\', 'ect', 'a2b', ram_cached=False)
    testing_data_loader = DataLoader(dataset=test_dataset, num_workers=1, batch_size=1,
                                     shuffle=False)
    # batch[0] (1,3,256,256)
    for i, batch in enumerate(testing_data_loader):
        print(batch[0][0].shape)
        real_b = batch[1] * 0.5 + 0.5
        real_a = batch[1] * 0.5 + 0.5
        torch.cat([real_a[0], real_b[0]], dim=2)
        tensor_to_PIL(batch[0][0]).show()
        tensor_to_PIL(real_b[0]).show()
        save_pil_image(tensor_to_PIL(batch[0][0]), r"C:\DATA\projects\dissertation\save_data\1.jpeg")
        if i == 0: break


def test5():
    train_dataset = get_train_dataset(r'/home/heng/DATA/datasets/pix2pix', 'ect-new', 'a2b', ram_cached=False)
    testing_data_loader = DataLoader(dataset=train_dataset, num_workers=1, batch_size=1,
                                     shuffle=False)
    for i, batch in enumerate(testing_data_loader):
        print(batch[0][0].shape)
        real_b = batch[1] * 0.5 + 0.5
        real_a = batch[0] * 0.5 + 0.5
        cat = torch.cat([real_a[0], real_b[0]], dim=2)
        save_pil_image(tensor_to_PIL(cat), r'/home/heng/DATA/projects/save_data/test.jpeg')
        if i == 0: break


if __name__ == "__main__":
    test5()
