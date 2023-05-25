from torchvision import transforms

unloader = transforms.ToPILImage()
loader = transforms.ToTensor()

def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image


def PIL_to_tensor(image):
    return loader(image)


