import numpy as np
from ..tensor import Tensolr

class Module:
    """Base class for neural network modules"""
    def __init__(self):
        self.training = True

    def forward(self, *input):
        raise NotImplementedError

    def __call__(self, *input):
        return self.forward(*input)

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self

    def parameters(self):
        """Return an iterator over module parameters"""
        for attr in dir(self):
            param = getattr(self, attr)
            if isinstance(param, Parameter):
                yield param
            elif isinstance(param, Module):
                yield from param.parameters()

class Parameter:
    """A tensor that is considered a module parameter"""
    def __init__(self, tensor):
        self.tensor = tensor

    def __getattr__(self, name):
        # Delegate attribute access to the tensor
        return getattr(self.tensor, name)

    def __repr__(self):
        return f"Parameter({self.tensor})"

class Linear(Module):
    """Linear (fully connected) layer"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights with Xavier initialization
        self.weight = Parameter(Tensolr.randn((in_features, out_features)) * np.sqrt(2.0 / (in_features + out_features)))
        
        if bias:
            self.bias = Parameter(Tensolr.zeros((out_features,)))
        else:
            self.bias = None

    def forward(self, x):
        output = x.matmul(self.weight.tensor)
        if self.bias is not None:
            output = output.add(self.bias.tensor)
        return output

class ReLU(Module):
    """ReLU activation function"""
    def forward(self, x):
        # Using the fact that x * (x > 0) gives ReLU
        zeros = Tensolr.zeros(x.shape)
        mask = x.sub(zeros)  # This will be positive where x > 0
        # For now, we'll use numpy comparison and create tensor
        result_data = np.maximum(0, x.data)
        result = Tensolr(result_data, track_graph=False)
        # Note: Since we're not implementing proper ReLU gradient here,
        # we'll just add it to the graph without proper gradient computation
        result._node = x._add_to_graph(result, "relu")
        return result

class Sigmoid(Module):
    """Sigmoid activation function"""
    def forward(self, x):
        # Sigmoid = 1 / (1 + exp(-x))
        ones = Tensolr.ones(x.shape)
        neg_x = x.mul(Tensolr([[-1]]))  # Assuming broadcasting works
        exp_neg_x = Tensolr(np.exp(-x.data), track_graph=False)
        denominator = ones.add(exp_neg_x)
        result = ones.div(denominator)
        result._node = x._add_to_graph(result, "sigmoid")
        return result

class Tanh(Module):
    """Tanh activation function"""
    def forward(self, x):
        # Tanh = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        exp_x = Tensolr(np.exp(x.data), track_graph=False)
        neg_x = x.mul(Tensolr([[-1]]))
        exp_neg_x = Tensolr(np.exp(-x.data), track_graph=False)
        numerator = exp_x.sub(exp_neg_x)
        denominator = exp_x.add(exp_neg_x)
        result = numerator.div(denominator)
        result._node = x._add_to_graph(result, "tanh")
        return result

class Sequential(Module):
    """Sequential container for stacking modules"""
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x):
        for module in self.modules:
            x = module(x)
        return x

    def parameters(self):
        for module in self.modules:
            yield from module.parameters()

class MSE(Module):
    """Mean Squared Error loss"""
    def forward(self, y_pred, y_true):
        diff = y_pred.sub(y_true)
        squared_diff = diff.mul(diff)
        # For now, we'll just return the mean along the first axis
        mean_squared_error = Tensolr(squared_diff.data.mean(axis=0), track_graph=False)
        mean_squared_error._node = y_pred._add_to_graph(mean_squared_error, "mse_loss")
        return mean_squared_error