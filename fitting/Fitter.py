import torch
import copy
import numpy as np

from fitting.helpers import generate_random_variants
from utils.image_helpers import image_to_tensor, tensor_to_image


def create_fitter_from_configuration(configuration):
    fitter = Fitter(number_of_iterations=configuration.number_of_iterations,
                    learning_rate=configuration.learning_rate,
                    convergence_check_length=configuration.convergence_check_length,
                    log_frequency=configuration.log_frequency,
                    find_best=configuration.find_best,
                    data_type=configuration.data_type,
                    save_losses=configuration.save_losses,
                    constant_fixed_input=configuration.constant_input,
                    compactness_parameter=configuration.compactness_parameter,
                    input_blur=configuration.input_blur,
                    lr_schedule_factor=configuration.lr_schedule_factor)
    return fitter


class Fitter:
    def __init__(self, number_of_iterations, learning_rate=0.01, convergence_check_length=None, log_frequency=10,
                 find_best=False, data_type=torch.FloatTensor, save_losses=False, constant_fixed_input=False,
                 compactness_parameter=0.9, input_blur=0, lr_schedule_factor=0.8):
        self.loss_function = torch.nn.MSELoss().type(data_type)
        self.number_of_iterations = number_of_iterations
        self.learning_rate = learning_rate
        self.convergence_check_length = convergence_check_length
        self.log_frequency = log_frequency
        self.find_best = find_best
        self.data_type = data_type
        self.save_losses = save_losses
        self.constant_fixed_input = constant_fixed_input
        self.fixed_net_input = None
        self.model = None
        self.save_steps = False
        self.compactness_parameter = compactness_parameter
        self.input_blur = input_blur
        self.lr_schedule_factor = lr_schedule_factor

    def __call__(self, model, original_image, target_image=None, log_prefix=None, loss_mask=None):
        if self.model is None or self.model == model:
            self.model = model.type(self.data_type)
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1000, self.lr_schedule_factor)
            self.best_model = copy.deepcopy(self.model)
            self.best_model_step = 0
            self.best_model_loss = 1000
            self.step_images = []
        self.step_counter = 0

        if (self.fixed_net_input is None) or (not self.constant_fixed_input):
            input_images = generate_random_variants(self.model.number_of_hidden_channels,
                                                    self.model.input_shape[-2:], blur=self.input_blur)
            fixed_net_input = torch.from_numpy(np.array(input_images)[None, :, :, :])
            self.fixed_net_input = fixed_net_input.type(self.data_type)
        self.noisy_image = image_to_tensor(original_image).type(self.data_type)

        if target_image is not None:
            self.target_image = image_to_tensor(target_image).type(self.data_type)
        else:
            self.target_image = None

        if loss_mask is None:
            self.loss_mask = torch.ones(self.noisy_image.shape).type(self.data_type)
        else:
            self.loss_mask = torch.from_numpy(loss_mask).squeeze().type(self.data_type)[None, :, :].type(self.data_type)

        if self.save_losses:
            self.losses_wrt_noisy = []
            self.losses_wrt_target = []
        self.current_loss_wrt_noisy = 1000
        self.current_loss_wrt_target = 1000
        if log_prefix is None:
            self.log_prefix = ''
        else:
            self.log_prefix = log_prefix
        self.fit()

    def fit(self):
        while self.has_not_converged() and self.step_counter < self.number_of_iterations:
            if False and self.step_counter > 1000:
                for module in self.model.module_list:
                    if type(module) == torch.nn.modules.dropout.Dropout:
                        module.eval()

            def closure():
                self.optimizer.zero_grad()
                output = self.model(self.fixed_net_input)
                loss = self.compactness_parameter * self.loss_function(self.noisy_image[-3:, :, :] * self.loss_mask[-3:, :, :],
                                                             output[-3:, :, :] * self.loss_mask[-3:, :, :])
                loss += (1 - self.compactness_parameter) * self.loss_function(
                    self.noisy_image[:-3, :, :] * self.loss_mask[:-3, :, :],
                    output[:-3, :, :] * self.loss_mask[:-3, :, :])
                loss.backward()
                self.update_loss_metrics_and_best_model(loss, output)
                self.scheduler.step()

            self.optimizer.step(closure)
            self.step_counter += 1

            if self.should_log():
                self.log()

            if self.step_counter == 1000:
                for module in self.model.module_list:
                    if type(module) == torch.nn.modules.dropout.Dropout:
                        module.p = 0.0

    def has_not_converged(self):
        if self.convergence_check_length is None:
            return True
        elif self.step_counter < self.convergence_check_length:
            return True
        else:
            if self.best_model_step < self.step_counter - self.convergence_check_length:
                print(self.log_prefix + f'Converged at step {self.step_counter}.' + ' ' * 50, end='\r')
                return False
        return True

    def update_loss_metrics_and_best_model(self, current_loss_wrt_noisy, current_output):
        self.model.eval()
        self.current_loss_wrt_noisy = current_loss_wrt_noisy.data

        if self.save_losses:
            self.losses_wrt_noisy.append(self.current_loss_wrt_noisy)

        if self.target_image is not None:
            current_loss_wrt_target = self.loss_function(current_output, self.target_image)
            self.current_loss_wrt_target = current_loss_wrt_target.data
            if self.save_losses:
                self.losses_wrt_target.append(self.current_loss_wrt_target.data)

        if self.find_best:
            if self.step_counter > 0:
                if self.best_model_loss > 1.005 * current_loss_wrt_noisy.data:
                    self.best_model = copy.deepcopy(self.model)
                    self.best_model_step = self.step_counter
                    self.best_model_loss = current_loss_wrt_noisy.data
        elif self.step_counter == self.number_of_iterations - 1:
            self.best_model = copy.deepcopy(self.model)
            for params, best_params in zip(self.model.named_parameters(), self.best_model.named_parameters()):
                best_params[1].grad = params[1].grad
            self.best_model_step = self.step_counter
            self.best_model_loss = current_loss_wrt_noisy.data

        self.model.train()

    def should_log(self):
        if self.step_counter % self.log_frequency == 0:
            return True
        elif self.step_counter == self.number_of_iterations:
            return True
        else:
            return False

    def log(self):
        log_string = self.log_prefix
        log_string += f"Step: {self.step_counter:05d}"
        log_string += ", "
        log_string += f"Loss: {self.current_loss_wrt_noisy:.6f}"
        if self.target_image is not None:
            log_string += ", "
            log_string += f"Target Loss: {self.current_loss_wrt_target:.6f}"
        if self.find_best:
            log_string += ', '
            log_string += f'Minimum Loss at: {self.best_model_step} with {self.best_model_loss:.6f}'
        print(log_string, end='\r')

    def get_best_image(self):
        return tensor_to_image(self.best_model(self.fixed_net_input).detach().cpu())
