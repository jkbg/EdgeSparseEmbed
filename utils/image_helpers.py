import numpy as np
import torch
import skimage
import skimage.measure

from skimage.segmentation import mark_boundaries


def tensor_to_image(tensor):
    """ (c,w,h) -> (w,h,c) """
    image = np.transpose(tensor.numpy(), (1, 2, 0))
    return image


def image_to_tensor(image):
    """ (w,h,c) -> (c,w,h) """
    tensor = torch.from_numpy(np.transpose(image, (2, 0, 1)))
    return tensor


def load_image(image_path):
    """ Load image and normalize to [0,1]. Add 3rd dimension if original is black and white."""
    original_image = skimage.io.imread(image_path)
    original_image = original_image.astype(float) / 255
    if len(original_image.shape) == 2:
        original_image = original_image[:, :, None]
    return original_image


def save_boundary_image(cluster_num, original_image, segmentation):
    if original_image.shape[2] == 1:
        original_image = skimage.color.gray2rgb(original_image[:, :, 0])
    display = mark_boundaries(original_image, segmentation, color=[1, 1, 0], outline_color=[1, 0, 0],
                              mode='subpixel')
    display = 255 * display
    display = display.astype(np.uint8)
    skimage.io.imsave(f'{cluster_num}_superpixels.png', display)
