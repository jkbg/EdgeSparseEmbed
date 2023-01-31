import os
import torch
import skimage

from skimage.segmentation import mark_boundaries

from FittingConfiguration import FittingConfiguration
from models.DeepDecoder import create_model_from_configuration
from fitting.Fitter import create_fitter_from_configuration
from fitting.helpers import create_sin_target
from utils.image_helpers import load_image
from utils.segmentation import segment, average_channel_gradients, assure_act_map_validity


# Backend Configuration
GPU = False
if GPU:
    assert torch.cuda.is_available()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print(torch.cuda.is_available(), "num GPUs", torch.cuda.device_count())
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor


def generate_superpixels(image_path, cluster_nums):
    """
    Parent function to generate superpixels.
    :param image_path: Path to image that gets superpixelized
    :param cluster_nums: List of cluster parameters, i.e. number of desired superpixels
    :return: None, images get saved as files
    """
    # Load Image
    original_image = load_image(image_path)

    # Setup Config with a sample positionally encoded target image (to set target image dimensions)
    sample_target_image = create_sin_target(original_image)
    fit_configuration = FittingConfiguration(sample_target_image)
    print(f'+++ Fitting Configuration +++')
    print(fit_configuration)
    print()

    # Init Decoders, Fitters and Target Images
    decoders = [create_model_from_configuration(fit_configuration) for _ in range(fit_configuration.number_of_decoders)]
    fitters = [create_fitter_from_configuration(fit_configuration) for _ in range(fit_configuration.number_of_decoders)]
    target_images = [create_sin_target(original_image) for _ in range(fit_configuration.number_of_decoders)]

    # Fit Decoders and extract activation maps

    last_layers = []
    for run_index, (decoder, fitter, target_image) in enumerate(zip(decoders, fitters, target_images)):
        # Fit
        fitter(decoder, target_image, log_prefix=f'Run {run_index + 1}: ')
        decoder = fitter.best_model

        # Extract Activation Maps

        # Generate Hook that saves the ouput of a module in the dictionary "activation"
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        # Register hook in last hidden layer, run decoder once and then save activation maps
        decoder.module_list[-3].register_forward_hook(get_activation('conv-1'))
        _ = decoder(fitter.fixed_net_input)
        last_layers.append(list(activation['conv-1'].squeeze().cpu().numpy()))

        print()

    # Assure Deep Decoders Converged & Channel Non-degeneracy
    selected_act_maps = assure_act_map_validity(fit_configuration, fitters, last_layers)

    channel_gradient = average_channel_gradients(selected_act_maps)

    # Iterate over number of clusters
    for cluster_num in cluster_nums:
        print(f'+++ SLIC - {cluster_num} Clusters +++' + ' ' * 20)
        segmentation = segment(selected_act_maps, cluster_num, channel_gradient)
        display = mark_boundaries(original_image, segmentation, color=[1, 1, 0], outline_color=[1, 0, 0],
                                  mode='subpixel')
        skimage.io.imsave(f'{cluster_num}_superpixels.jpg', display)
        print()


if __name__ == '__main__':
    cluster_nums = [20, 100, 400]
    image_path = 'breakfast.png'
    generate_superpixels(image_path, cluster_nums)
