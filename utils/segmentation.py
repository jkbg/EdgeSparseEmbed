import numpy as np
from scipy.ndimage import binary_dilation
from sklearn.preprocessing import StandardScaler

from utils.slic import SLIC


def segment(filtered_act_maps, number_of_clusters, channel_gradient):
    """ Parent function of the clustering step. """
    # Build Feature Matrix & Apply Standard Scaling
    maps = np.array(filtered_act_maps)
    coordinates = np.argwhere(np.ones(shape=filtered_act_maps[0].shape).astype(bool))
    feature_matrix = np.zeros((coordinates.shape[0], maps[:, 0, 0].size))
    for pixel_index, (x, y) in enumerate(coordinates):
        feature_matrix[pixel_index, :] = maps[:, x, y]
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(feature_matrix)
    pixel_features = feature_matrix.reshape(*filtered_act_maps[0].shape, -1)

    # Apply custom SLIC
    slic = SLIC(number_of_clusters, pixel_features, channel_gradient)
    segmentation = slic.iterate(100)

    return segmentation


def average_channel_gradients(filtered_act_maps):
    """ Finds boundaries based on ReLu cuts in each channel
    Return the mean of these boundary maps
    """
    channel_gradients = []
    for act_map in filtered_act_maps:
        bin_edge = binary_dilation(act_map == act_map.min()) & binary_dilation(act_map != act_map.min())
        channel_gradients.append(bin_edge)
    channel_gradient = np.mean(channel_gradients, axis=0)
    return channel_gradient


def is_usable(act_map):
    # Check for degeneracy
    if act_map.max() - act_map.min() > 0.005:
        if act_map.max() * act_map.min() < 0:
            return True
    return False


def assure_act_map_validity(fit_configuration, fitters, last_layers):
    filtered_act_maps = []
    # Only save activation maps of deep decoders that showed stable convergence
    # i.e. their best model step happened sufficiently late
    # And discard degenerate channels
    for fitter, last_layer in zip(fitters, last_layers):
        if fitter.best_model_step > 0.75 * fit_configuration.number_of_iterations:
            filtered_act_maps.extend([act_map for act_map in last_layer if is_usable(act_map)])
    if len(filtered_act_maps) == 0:
        best_steps = [fitter.best_model_step for fitter in fitters]
        filtered_act_maps = last_layers[np.argmax(best_steps)]
    return filtered_act_maps