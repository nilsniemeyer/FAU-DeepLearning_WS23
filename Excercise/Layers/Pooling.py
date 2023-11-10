from . import Base
import numpy as np
from icecream import ic


class Pooling(Base.BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_y, self.stride_x = stride_shape
        self.pooling_shape = pooling_shape
        self.pooling_y, self.pooling_x = pooling_shape
        self.input_tensor = None
        self.output_tensor = None
        self.maxima_positions = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_size, channel, input_height, input_width = input_tensor.shape

        # Calculate output dimensions
        out_height = 1 + (input_height - self.pooling_y) // self.stride_y  # One plus the possible strides
        out_width = 1 + (input_width - self.pooling_x) // self.stride_x

        self.output_tensor = np.zeros((batch_size, channel, out_height, out_width))
        self.maxima_positions = np.empty_like(self.output_tensor, dtype=object)

        for i in range(batch_size):
            for j in range(channel):
                batch_item = input_tensor[i, j, :, :]
                # Defining windows based on the position in the output array (This avoids using numpy's stride_tricks.as_strided)
                for y in range(out_height):
                    for x in range(out_width):
                        y_start = y * self.stride_y
                        x_start = x * self.stride_x
                        window = batch_item[y_start: y_start + self.pooling_y, x_start: x_start + self.pooling_x]
                        max = np.max(window)
                        window_max_y, window_max_x = np.unravel_index(np.argmax(window), (self.pooling_y, self.pooling_x))
                        max_y = y_start + window_max_y
                        max_x = x_start + window_max_x
                        self.output_tensor[i, j, y, x] = max
                        self.maxima_positions[i, j, y, x] = (max_y, max_x)
        return self.output_tensor


    def backward(self, error_tensor):
        # Initialize the error tensor for the previous layer
        batch_size, channel, input_height, input_width = self.input_tensor.shape
        error_to_previous_layer = np.zeros_like(self.input_tensor)

        # Accumulate the gradients at the positions of the maxima
        for i in range(batch_size):
            for j in range(channel):
                for y in range(error_tensor.shape[2]):
                    for x in range(error_tensor.shape[3]):
                        # Get the position of the maximum value in the original window
                        max_y, max_x = self.maxima_positions[i, j, y, x]

                        # Accumulate the gradient at the position of the maximum value
                        error_to_previous_layer[i, j, max_y, max_x] += error_tensor[i, j, y, x]

        return error_to_previous_layer