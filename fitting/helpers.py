import numpy as np
import cv2
import skimage


def generate_random_variants(number_of_hidden_channels, input_size, blur=0):
    input_images = []
    gen_counter = 0

    while len(input_images) < number_of_hidden_channels and gen_counter < number_of_hidden_channels * 5:
        noise = np.random.uniform(low=-1, high=1, size=[*input_size])
        if blur == 0:
            convolution = noise
        else:
            convolution = cv2.GaussianBlur(noise, (blur, blur), 0)
            convolution = convolution - np.min(convolution)
            convolution = convolution / np.max(convolution)
            convolution = 2 * convolution - 1

        input_images.append(convolution)

        gen_counter += 1
    for _ in range(number_of_hidden_channels - len(input_images)):
        input_images.append(np.zeros([input_size] * 2))
    return input_images


def create_sin_target(original_image):
    clab_img = skimage.color.rgb2lab(original_image)
    clab_img[:, :, 0] = clab_img[:, :, 0] / 100
    clab_img[:, :, 1] = (clab_img[:, :, 1] + 128) / 255
    clab_img[:, :, 2] = (clab_img[:, :, 2] + 128) / 255

    target_image = np.zeros((original_image.shape[0], original_image.shape[1], 0))
    xs = np.arange(original_image.shape[0])
    ys = np.arange(original_image.shape[1])
    yy, xx = np.meshgrid(ys, xs)
    norm = max(original_image.shape[0], original_image.shape[1])
    xx = xx / norm
    yy = yy / norm

    encodings = []
    for l in [1, 2, 3]:
        for zz in [xx, yy]:
            for _ in range(1):
                offset = np.random.uniform(0,2*np.pi)
                encodings.append(np.sin(2 ** l * np.pi * zz + offset))
                encodings.append(np.sin(2 ** l * np.pi * zz + offset + np.pi))
                offset = np.random.uniform(0,2*np.pi)
                encodings.append(np.cos(2 ** l * np.pi * zz + offset))
                encodings.append(np.cos(2 ** l * np.pi * zz + offset + np.pi))
    encodings = [encoding for encoding in encodings if np.any(encoding > 0.001)]
    for encoding in encodings:
        encoding = 0.5 * encoding + 0.5
        new_channel = encoding * clab_img[:, :, 0]
        target_image = np.concatenate((target_image, new_channel[:, :, None]), axis=2)
    target_image = np.concatenate((target_image, original_image), axis=2)
    return target_image
