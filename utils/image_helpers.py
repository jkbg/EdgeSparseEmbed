import numpy as np
import torch
import skimage
import skimage.measure

from fitting.helpers import create_sin_target


def tensor_to_image(tensor):
    image = np.transpose(tensor.numpy(), (1, 2, 0))
    return image


def image_to_tensor(image):
    tensor = torch.from_numpy(np.transpose(image, (2, 0, 1)))
    return tensor


def load_image(image_path):
    original_image = skimage.io.imread(image_path)
    original_image = original_image.astype(float) / 255
    if len(original_image.shape) == 2:
        original_image = original_image[:, :, None]
    return original_image, create_sin_target(original_image)