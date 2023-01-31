import torch
import os


class FittingConfiguration:
    def __init__(self, target_image):
        self.image_dimensions = target_image.shape
        self.result_path = None

        # Architecture Parameters
        self.model_type = 'deep'
        self.input_shape = [target_image.shape[0] // 6, target_image.shape[1] // 6]
        self.number_of_layers = 5
        self.number_of_hidden_channels = 128

        # Fitting Parameters
        self.number_of_iterations = 1500
        self.number_of_decoders = 3
        self.learning_rate = 0.1
        self.convergence_check_length = None
        self.log_frequency = 100
        self.find_best = True
        self.save_losses = False
        self.compactness_parameter = 0.1
        blur_factor = 0.0001
        self.input_blur = int(target_image[:, :, 0].size * blur_factor // 2 * 2 + 1)
        self.lr_schedule_factor = 0.8

        if torch.cuda.is_available():
            self.data_type = torch.cuda.FloatTensor
        else:
            self.data_type = torch.FloatTensor

    def __str__(self):
        dictionary = self.__dict__
        result = ""
        for key in dictionary:
            result += key + ": " + str(dictionary[key]) + "  " + os.linesep
        return result
