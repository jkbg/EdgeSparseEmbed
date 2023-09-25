import torch
import copy
import numpy as np

from fitting.helpers import generate_random_input
from utils.image_helpers import image_to_tensor, tensor_to_image


def create_fitter_from_configuration(configuration):
    """
    Takes a fit configuration object and returns a fitter.
    :param configuration: FitConfiguration
    :return: Fitter
    """
    fitter = Fitter(number_of_iterations=configuration.number_of_iterations,
                    learning_rate=configuration.learning_rate,
                    convergence_check_length=configuration.convergence_check_length,
                    log_frequency=configuration.log_frequency,
                    find_best=configuration.find_best,
                    data_type=configuration.data_type,
                    save_losses=configuration.save_losses,
                    compactness_parameter=configuration.compactness_parameter,
                    input_blur=configuration.input_blur,
                    lr_schedule_factor=configuration.lr_schedule_factor)
    return fitter


class Fitter:
    """ Object that when called iterates through training loop """
    def __init__(self, number_of_iterations, learning_rate=0.01, convergence_check_length=None, log_frequency=10,
                 find_best=False, data_type=torch.FloatTensor, save_losses=False,
                 compactness_parameter=0.9, input_blur=0, lr_schedule_factor=0.8):
        self.loss_function = torch.nn.MSELoss().type(data_type)
        self.number_of_iterations = number_of_iterations
        self.learning_rate = learning_rate
        self.convergence_check_length = convergence_check_length
        self.log_frequency = log_frequency
        self.find_best = find_best
        self.data_type = data_type
        self.save_losses = save_losses
        self.fixed_net_input = None
        self.model = None
        self.save_steps = False
        self.compactness_parameter = compactness_parameter
        self.input_blur = input_blur
        self.lr_schedule_factor = lr_schedule_factor

    def __call__(self, model, target_image, log_prefix=None):
        """
        First initializes optimizer, scheduler, etc.
        Then calls self.fit()

        :param model: DeepDecoder
        :param target_image: w*h*c numpy array
        :param log_prefix: str
        :return:
        """
        self.model = model.type(self.data_type)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1000, self.lr_schedule_factor)
        self.best_model = copy.deepcopy(self.model)
        self.best_model_step = 0
        self.best_model_loss = 1000
        self.step_images = []
        self.step_counter = 0
        self.target_image = image_to_tensor(target_image).type(self.data_type)
        if self.save_losses:
            self.losses_wrt_target = []
        self.current_loss_wrt_target = 1000
        if log_prefix is None:
            self.log_prefix = ''
        else:
            self.log_prefix = log_prefix

        # If no net_input was specified, generate model input
        if self.fixed_net_input is None:
            input_images = generate_random_input(self.model.number_of_hidden_channels,
                                                    self.model.input_shape[-2:], blur=self.input_blur)
            fixed_net_input = torch.from_numpy(np.array(input_images)[None, :, :, :])
            self.fixed_net_input = fixed_net_input.type(self.data_type)

        # Start fitting loop
        self.fit()

    def fit(self):
        while self.has_not_converged() and self.step_counter < self.number_of_iterations:
            def closure():
                self.optimizer.zero_grad()
                output = self.model(self.fixed_net_input)
                # Calculate loss wrt RGB
                loss = (1-self.compactness_parameter) * self.loss_function(self.target_image[24:, :, :], output[24:, :, :])
                # Calculate loss wrt positional encoding
                loss += self.compactness_parameter * self.loss_function(self.target_image[:24, :, :], output[:24, :, :])
                loss.backward()
                self.update_loss_metrics_and_best_model(loss)
                self.scheduler.step()

            self.optimizer.step(closure)
            self.step_counter += 1

            if self.should_log():
                self.log()

            # Set dropout to 0 after 1000 steps.
            if self.step_counter == 1000:
                for module in self.model.module_list:
                    if type(module) == torch.nn.modules.dropout.Dropout:
                        module.p = 0.0

    def has_not_converged(self):
        """ If self.convergence_check_length specified it checks whether the best loss has improved within the last
        self.convergence_check_length steps
        """
        if self.convergence_check_length is None:
            return True
        elif self.step_counter < self.convergence_check_length:
            return True
        else:
            if self.best_model_step < self.step_counter - self.convergence_check_length:
                print(self.log_prefix + f'Converged at step {self.step_counter}.' + ' ' * 50, end='\r')
                return False
        return True

    def update_loss_metrics_and_best_model(self, current_loss_wrt_target):
        """ Saves current loss wrt target and updates self.best_model if loss decreased significantly """
        self.current_loss_wrt_target = current_loss_wrt_target.data

        if self.save_losses:
            self.losses_wrt_target.append(self.current_loss_wrt_target)


        if self.find_best:
            if self.step_counter > 0:
                if self.best_model_loss > 1.005 * current_loss_wrt_target.data:
                    self.best_model = copy.deepcopy(self.model)
                    self.best_model_step = self.step_counter
                    self.best_model_loss = current_loss_wrt_target.data
        elif self.step_counter == self.number_of_iterations - 1:
            self.best_model = copy.deepcopy(self.model)
            self.best_model_step = self.step_counter
            self.best_model_loss = current_loss_wrt_target.data

    def should_log(self):
        """ Checks if losses should be logged based on self.step_counter. """
        if self.step_counter % self.log_frequency == 0:
            return True
        elif self.step_counter == self.number_of_iterations:
            return True
        else:
            return False

    def log(self):
        """ Prints losses and best model loss and step. """
        log_string = self.log_prefix
        log_string += f"Step: {self.step_counter:05d}"
        log_string += ", "
        log_string += f"Loss: {self.current_loss_wrt_target:.6f}"
        if self.find_best:
            log_string += ', '
            log_string += f'Minimum Loss at: {self.best_model_step} with {self.best_model_loss:.6f}'
        print(log_string, end='\r')

    def get_best_image(self):
        """ Returns reconstruction of best model. """
        return tensor_to_image(self.best_model(self.fixed_net_input).detach().cpu())
