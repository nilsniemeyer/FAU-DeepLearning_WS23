import numpy as np
from scipy.signal import correlate, convolve, resample
from . import Base
from . import Initializers


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
        # 1D and 2D convolution certainly could be combined, but this should do for now. Only 2D is commented.
        if input_tensor.ndim == 3:
            # 1D
            batch_size, input_channels, input_length = input_tensor.shape

            output_length = (input_length+self.stride_x-1) // self.stride_x
            output_tensor = np.zeros((batch_size, self.num_kernels, output_length))

            for batch_index in range(batch_size):
                for kernel_index in range(self.num_kernels):
                    kernel_output = []
                    for channel_index in range(input_channels):
                        channel_input = input_tensor[batch_index, channel_index, :]
                        channel_kernel = self.weights[kernel_index, channel_index, :]
                        cross_correlation = correlate(channel_input, channel_kernel, mode='same', method='direct')
                        kernel_output.append(cross_correlation[::self.stride_x])
                    output_tensor[batch_index, kernel_index, :] = np.sum(np.stack(kernel_output, axis=0), axis=0) + self.bias[kernel_index]

        elif input_tensor.ndim == 4:
            # 2D
            batch_size, input_channels, input_height, input_width = input_tensor.shape
            # Determine output shape to create an output tensor, which is later filled with the output of the convolution
            output_height = (input_height+self.stride_y-1) // self.stride_y
            output_width = (input_width+self.stride_x-1) // self.stride_x
            output_tensor = np.zeros((batch_size, self.num_kernels, output_height, output_width))

            for batch_index in range(batch_size):  # For every element in the batch
                # The idea is to slice the input and kernels layer by layer, cross-correlate them and sum them up to get the output layers
                for kernel_index in range(self.num_kernels):  # For every kernel (basically creating the output layer one by one)
                    kernel_output = []  # List to store the cross-correlations for all input channels. These are later summed up into one output layer
                    for channel_index in range(input_channels):  # For every input channel (Going through the input layer one by one. No 3D operation.)
                        channel_input = input_tensor[batch_index, channel_index, :, :]  # Get the channel of the input
                        channel_kernel = self.weights[kernel_index, channel_index, :, :]  # Get the channel of the kernel
                        cross_correlation = correlate(channel_input, channel_kernel, mode='same', method='direct')  # Cross-correlate the input and kernel
                        kernel_output.append(cross_correlation[::self.stride_y, ::self.stride_x])  # Append the cross-correlation to the list
                    # Stack the cross-correlations of the input channels and sum them up. Add the bias. Save the output layer into the output tensor.
                    output_tensor[batch_index, kernel_index, :, :] = np.sum(np.stack(kernel_output, axis=0), axis=0) + self.bias[kernel_index]

        else:
            raise ValueError("Invalid convolution size.")

        self.output_tensor = output_tensor

        return output_tensor

    def backward(self, error_tensor):
        # 1D and 2D convolution certainly could be combined, but this should do for now. Only 2D is commented.
        # The error tensor for the previous layer has the same shape as the input tensor of this layer.
        prev_error_tensor = np.zeros_like(self.input_tensor)
        if error_tensor.ndim == 3:
            # 1D
            # Calculate gradient for input tensor:
            for batch_index in range(error_tensor.shape[0]):
                reshaped_weights = np.transpose(self.weights.copy(), (1, 0, 2))
                for input_channel in range(reshaped_weights.shape[0]):
                    prev_channel_error = []
                    for old_kernel_layer in range(reshaped_weights.shape[1]):
                        kernel_layer = reshaped_weights[input_channel, old_kernel_layer, :]

                        error_layer = error_tensor[batch_index, old_kernel_layer, :]
                        upscaled_error_layer = np.zeros_like(self.input_tensor[batch_index, input_channel, :])
                        upscaled_error_layer[::self.stride_x] = error_layer

                        convolution = convolve(upscaled_error_layer, kernel_layer, mode='same', method='direct')
                        prev_channel_error.append(convolution)
                    prev_error_tensor[batch_index, input_channel, :] = np.sum(np.stack(prev_channel_error, axis=0), axis=0)

            # Calculate gradient for weights:
            self.gradient_weights = np.zeros_like(self.weights)

            for batch_index in range(error_tensor.shape[0]):
                for kernel_index in range(error_tensor.shape[1]):
                    for channel_index in range(self.input_tensor.shape[1]):
                        error_layer = error_tensor[batch_index, kernel_index, :]
                        upscaled_error_layer = np.zeros_like(self.input_tensor[batch_index, channel_index, :])
                        upscaled_error_layer[::self.stride_x] = error_layer

                        input_layer = self.input_tensor[batch_index, channel_index, :]
                        total_pad_width = self.convolution_shape[1] - 1
                        pad_left = total_pad_width // 2
                        pad_right = total_pad_width - pad_left
                        padded_input_layer = np.pad(input_layer, ((pad_left, pad_right)), mode='constant', constant_values=0)

                        cross_correlation = correlate(padded_input_layer, upscaled_error_layer, mode='valid', method='direct')
                        self.gradient_weights[kernel_index, channel_index, :] += cross_correlation

            # Calculate gradient for bias:
            self.gradient_bias = np.sum(error_tensor, axis=(0, 2)).reshape(-1, 1)

        elif error_tensor.ndim == 4:
            # 2D
            # Calculate gradient for input tensor
            for batch_index in range(error_tensor.shape[0]):
                # The idea is to rearrange the kernels. Then, we go through the input layers, which now each have a respective kernel.
                reshaped_weights = np.transpose(self.weights.copy(), (1, 0, 2, 3))
                for input_channel in range(reshaped_weights.shape[0]):
                    # For every input channel, we go through every layer of the reshaped kernel (which corresponds to different original kernels)
                    # Each of these layers has a respective error layer. We go through these corresponding layers one by one, stack the results and sum them up.
                    prev_channel_error = []  # List to store the convolutions for all kernel and output layers of this input layer. These are later summed up into one.
                    for old_kernel_layer in range(reshaped_weights.shape[1]):
                        kernel_layer = reshaped_weights[input_channel, old_kernel_layer, :, :]  # Get the kernel layer

                        error_layer = error_tensor[batch_index, old_kernel_layer, :, :]  # Get the error layer
                        upscaled_error_layer = np.zeros_like(self.input_tensor[batch_index, input_channel, :, :])  # Create an error layer with the same shape as the input layer (used for upsampling because of stride in the forward pass)
                        upscaled_error_layer[::self.stride_y, ::self.stride_x] = error_layer  # Fill the error layer with the error values, but keeping the skipped places (bc stride) at 0 (zero interpolation)

                        convolution = convolve(upscaled_error_layer, kernel_layer, mode='same', method='direct')  # Convolve the error layer with the kernel layer
                        prev_channel_error.append(convolution)
                    prev_error_tensor[batch_index, input_channel, :, :] = np.sum(np.stack(prev_channel_error, axis=0), axis=0)  # Stack and sum the convolutions to get the error tensor for this input layer.

            # Calculate gradient for weights:
            self.gradient_weights = np.zeros_like(self.weights)  # Resetting the gradient weights to zero (Because we accumulate the gradients)

            for batch_index in range(error_tensor.shape[0]):
                # The idea is to go through every error layer (or kernel) and cross-correlate it with each input layer (channel) to get gradients for each layer of the kernel.
                for kernel_index in range(error_tensor.shape[1]):
                    for channel_index in range(self.input_tensor.shape[1]):
                        error_layer = error_tensor[batch_index, kernel_index, :, :]  # Get the error layer for this kernel
                        upscaled_error_layer = np.zeros_like(self.input_tensor[batch_index, channel_index, :, :])  # Create an error layer with the same shape as the input layer (used for upsampling because of stride in the forward pass)
                        upscaled_error_layer[::self.stride_y, ::self.stride_x] = error_layer  # Fill the error layer with the error values, but keeping the skipped places (bc stride) at 0 (zero interpolation)

                        # Now we need to pad the input tensor. Currently, both input tensor and error tensor have the same shape.
                        # We need the cross-correlation to result in the shape of the kernel.
                        # Therefore, we need to pad the input tensor, such that the error tensor can be shifted as often as the kernel shape dictates.
                        input_layer = self.input_tensor[batch_index, channel_index, :, :]  # Get the input channel for this layer of the kernel
                        # Because we also have even kernels, we need to allow for asymmetric padding.
                        total_pad_height = self.convolution_shape[1] - 1
                        total_pad_width = self.convolution_shape[2] - 1
                        pad_top = total_pad_height // 2
                        pad_bottom = total_pad_height - pad_top
                        pad_left = total_pad_width // 2
                        pad_right = total_pad_width - pad_left
                        padded_input_layer = np.pad(input_layer, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)

                        # Now we can simply (valid) cross-correlate the padded input layer with the error layer and accumulate this gradient.
                        # This is done for every layer of every kernel.
                        cross_correlation = correlate(padded_input_layer, upscaled_error_layer, mode='valid', method='direct')
                        self.gradient_weights[kernel_index, channel_index, :, :] += cross_correlation

            # Calculate gradient for bias:
            self.gradient_bias = np.sum(error_tensor, axis=(0, 2, 3)).reshape(-1, 1)  # The gradient wrt bias is simply the sum of the error tensor over all batch elements and all spatial dimensions.

        else:
            raise ValueError("Invalid convolution size.")

        # Update:
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        if self._bias_optimizer:
            self.bias = self._bias_optimizer.calculate_update(self.bias, self._gradient_bias)

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
