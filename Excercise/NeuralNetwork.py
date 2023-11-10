import copy


class NeuralNetwork():
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.input_tensor = None
        self.label_tensor = None

    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()
        # Passing the input data through each layer in the network, updating the input tensor.
        for layer in self.layers:
            self.input_tensor = layer.forward(self.input_tensor)
        # Returning the loss using the loss layer and the predicted output and labels
        return self.loss_layer.forward(self.input_tensor, self.label_tensor)

    def backward(self):
        error = self.loss_layer.backward(self.label_tensor)
        # Passing the error tensor backward through each layer in reverse order, updating the error tensor.
        for layer in reversed(self.layers):
            error = layer.backward(error)

    def append_layer(self, layer):
        # Creating a deepcopy of the optimizer, if the layer is trainable.Assigning it to the layer.
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def train(self, iterations):
        # Performing forward and backward propagation for each iteration and appending the loss.
        for i in range(iterations):
            print("iteration ", i)
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self, input_tensor):
        # Propagating the input tensor through each layer in the network and return the predicted output.
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor
