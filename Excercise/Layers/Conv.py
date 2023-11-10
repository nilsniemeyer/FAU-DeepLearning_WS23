import numpy as np
from scipy.signal import correlate, convolve, resample
from . import Base
from . import Initializers
from icecream import ic

class Conv(Base.BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True

        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.stride_y, self.stride_x = (None, None)

        # Initialize weights and bias such that a shapes are known for later. And set stride in x and y direction
        if len(self.convolution_shape) == 2:
            # 1D convolution
            self.weights = np.zeros((self.num_kernels, self.convolution_shape[0], self.convolution_shape[1]))
            self.bias = np.zeros(self.num_kernels)
            self.stride_x = stride_shape[0]

        elif len(self.convolution_shape) == 3:
            # 2D convolution
            self.weights = np.zeros((self.num_kernels, self.convolution_shape[0], self.convolution_shape[1], self.convolution_shape[2]))
            self.bias = np.zeros((self.num_kernels, 1))
            if isinstance(stride_shape, tuple):
                self.stride_y, self.stride_x = stride_shape
            else:
                self.stride_y = self.stride_x = stride_shape

        else:
            raise ValueError("Invalid convolution size.")

        # Actually initialize weights
        weights_initializer = Initializers.UniformRandom()
        bias_initializer = Initializers.UniformRandom()
        self.initialize(weights_initializer, bias_initializer)

        self.input_tensor = None
        self._gradient_weights = np.zeros_like(self.weights)
        self._gradient_bias = None
        self._optimizer = None
        self._bias_optimizer = None

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape, np.prod(self.convolution_shape), self.num_kernels * np.prod(self.convolution_shape[1:]))
        self.bias = bias_initializer.initialize(self.bias.shape, 1, self.num_kernels)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        if input_tensor.ndim == 3:
            # 1D
            batch_size, self.input_channels, input_length = input_tensor.shape

            output_length = (input_length+self.stride_x-1) // self.stride_x
            output_tensor = np.zeros((batch_size, self.num_kernels, output_length))

            self.padding_low_or_uneven = (self.convolution_shape[1] - 1) // 2
            self.padding_high = self.convolution_shape[1] // 2

            for batch_index in range(batch_size):
                b_input = input_tensor[batch_index, :, :]
                if self.convolution_shape[1] % 2 == 0:
                    b_input = np.pad(b_input, ((0, 0), (self.padding_low_or_uneven, self.padding_high)), 'constant', constant_values=0.0)
                else:
                    b_input = np.pad(b_input, ((0, 0), (self.padding_low_or_uneven, self.padding_low_or_uneven)), 'constant', constant_values=0.0)

                for kernel_index in range(self.num_kernels):
                    kernel = self.weights[kernel_index]
                    cross_correlation = correlate(b_input, kernel, mode='valid', method='direct')

                    subsampled_output = cross_correlation[:, ::self.stride_x]
                    output_tensor[batch_index, kernel_index] = subsampled_output
                    output_tensor[batch_index, kernel_index] += self.bias[kernel_index]

        elif input_tensor.ndim == 4:
            # 2D
            batch_size, self.input_channels, input_height, input_width = input_tensor.shape

            output_height = (input_height+self.stride_y-1) // self.stride_y
            output_width = (input_width+self.stride_x-1) // self.stride_x
            output_tensor = np.zeros((batch_size, self.num_kernels, output_height, output_width))

            self.height_padding_low_or_uneven = (self.convolution_shape[1] - 1) // 2
            self.height_padding_high = self.convolution_shape[1] // 2
            self.width_padding_low_or_uneven = (self.convolution_shape[2] - 1) // 2
            self.width_padding_high = self.convolution_shape[2] // 2

            for batch_index in range(batch_size):
                b_input = input_tensor[batch_index, :, :, :]
                if self.convolution_shape[1] % 2 == 0:
                    b_input = np.pad(b_input, ((0, 0), (self.height_padding_low_or_uneven, self.height_padding_high), (0, 0)), 'constant', constant_values=0.0)
                else:
                    b_input = np.pad(b_input, ((0, 0), (self.height_padding_low_or_uneven, self.height_padding_low_or_uneven), (0, 0)), 'constant', constant_values=0.0)
                if self.convolution_shape[2] % 2 == 0:
                    b_input = np.pad(b_input, ((0, 0), (0, 0), (self.width_padding_low_or_uneven, self.width_padding_high)), 'constant', constant_values=0.0)
                else:
                    b_input = np.pad(b_input, ((0, 0), (0, 0), (self.width_padding_low_or_uneven, self.width_padding_low_or_uneven)), 'constant', constant_values=0.0)
                for kernel_index in range(self.num_kernels):
                    kernel = self.weights[kernel_index]
                    cross_correlation = correlate(b_input, kernel, mode='valid', method='direct')
                    subsampled_output = cross_correlation[:, ::self.stride_y, ::self.stride_x]
                    output_tensor[batch_index, kernel_index] = subsampled_output
                    output_tensor[batch_index, kernel_index] += self.bias[kernel_index]
        else:
            raise ValueError("Invalid convolution size.")

        self.output_tensor = output_tensor

        return output_tensor

    def backward(self, error_tensor):
        ic(error_tensor.shape)
        prev_error_tensor = np.zeros_like(self.input_tensor)
        # Not implemented
        '''
        if error_tensor.ndim == 3:
            raise ValueError("Not implemented.")

        elif error_tensor.ndim == 4:
            # 2D
            for batch_index in range(error_tensor.shape[0]):
                b_error_tensor = error_tensor[batch_index, :, :, :]
                b_input_tensor = self.input_tensor[batch_index, :, :, :]
                ic(b_error_tensor.shape)
                ic(b_input_tensor.shape)
                reshaped_weights = np.transpose(self.weights.copy(), (1, 0, 2, 3))

        else:
            raise ValueError("Invalid convolution size.")
        '''
        return prev_error_tensor

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, value):
        self._gradient_bias = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def bias_optimizer(self):
        return self._bias_optimizer
    @bias_optimizer.setter
    def bias_optimizer(self, value):
        self._bias_optimizer = value
