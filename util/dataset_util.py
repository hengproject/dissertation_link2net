import os
from PIL import Image as Image
from util.visualize_util import PIL_to_tensor

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_resized_img(path):
    return PIL_to_tensor(Image.open(path).convert('RGB').resize((256, 256)))
