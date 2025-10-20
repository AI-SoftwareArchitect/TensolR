# Tensolr Documentation

## Overview

Tensolr is a tensor framework that provides similar functionality to TensorFlow. This documentation covers all the API classes and methods available in the framework.

## API Reference

### Core Tensor Class

#### `Tensolr`
The main tensor class that provides all the basic operations.

**Constructor:**
```python
Tensolr(data, track_graph=True)
```

**Arguments:**
- `data`: The input data, can be a list, numpy array, or other array-like structure
- `track_graph`: Whether to track the computational graph for automatic differentiation (default: True)

**Attributes:**
- `data`: The underlying numpy array containing the tensor data
- `shape`: The shape of the tensor
- `dtype`: The data type of the tensor
- `size`: The number of elements in the tensor
- `ndim`: The number of dimensions of the tensor

**Factory Methods:**
- `zeros(shape)`: Creates a tensor filled with zeros
- `ones(shape)`: Creates a tensor filled with ones
- `randn(shape)`: Creates a tensor filled with random numbers from a normal distribution

**Example:**
```python
import tensolr
tensor = tensolr.Tensolr([[1, 2], [3, 4]])
print(tensor.shape)  # (2, 2)
```

---

### Tensor Operations

#### `add(other)`
Performs element-wise addition with another tensor.

**Arguments:**
- `other`: Another Tensolr object to add

**Returns:**
- A new Tensolr object representing the sum

**Example:**
```python
a = Tensolr([[1, 2], [3, 4]])
b = Tensolr([[5, 6], [7, 8]])
c = a.add(b)  # [[6, 8], [10, 12]]
```

#### `sub(other)`
Performs element-wise subtraction with another tensor.

**Arguments:**
- `other`: Another Tensolr object to subtract

**Returns:**
- A new Tensolr object representing the difference

**Example:**
```python
a = Tensolr([[1, 2], [3, 4]])
b = Tensolr([[5, 6], [7, 8]])
c = a.sub(b)  # [[-4, -4], [-4, -4]]
```

#### `mul(other)`
Performs element-wise multiplication with another tensor.

**Arguments:**
- `other`: Another Tensolr object to multiply

**Returns:**
- A new Tensolr object representing the product

**Example:**
```python
a = Tensolr([[1, 2], [3, 4]])
b = Tensolr([[5, 6], [7, 8]])
c = a.mul(b)  # [[5, 12], [21, 32]]
```

#### `div(other)`
Performs element-wise division with another tensor.

**Arguments:**
- `other`: Another Tensolr object to divide by

**Returns:**
- A new Tensolr object representing the quotient

**Example:**
```python
a = Tensolr([[1, 2], [3, 4]])
b = Tensolr([[2, 2], [2, 2]])
c = a.div(b)  # [[0.5, 1.0], [1.5, 2.0]]
```

#### `pow(other)`
Performs element-wise exponentiation with another tensor.

**Arguments:**
- `other`: Another Tensolr object to use as exponents

**Returns:**
- A new Tensolr object representing the power

**Example:**
```python
a = Tensolr([[2, 3], [4, 5]])
b = Tensolr([[1, 2], [1, 2]])
c = a.pow(b)  # [[2, 9], [4, 25]]
```

#### `matmul(other)`
Performs matrix multiplication with another tensor.

**Arguments:**
- `other`: Another Tensolr object to multiply

**Returns:**
- A new Tensolr object representing the matrix product

**Example:**
```python
a = Tensolr([[1, 2], [3, 4]])
b = Tensolr([[5, 6], [7, 8]])
c = a.matmul(b)  # [[19, 22], [43, 50]]
```

#### `transpose()`
Transposes the tensor.

**Returns:**
- A new Tensolr object representing the transpose

**Example:**
```python
a = Tensolr([[1, 2], [3, 4]])
b = a.transpose()  # [[1, 3], [2, 4]]
```

---

### Neural Network API

#### `Module`
Base class for neural network modules.

**Methods:**
- `forward(*input)`: Defines the forward pass, should be implemented in subclasses
- `train()`: Sets the module in training mode
- `eval()`: Sets the module in evaluation mode
- `parameters()`: Returns an iterator over module parameters

**Example:**
```python
class MyModule(Module):
    def __init__(self):
        super().__init__()
        self.my_param = Parameter(Tensolr([1, 2, 3]))
    
    def forward(self, x):
        return x
```

#### `Parameter`
Wraps a tensor to indicate it's a module parameter.

**Constructor:**
```python
Parameter(tensor)
```

**Arguments:**
- `tensor`: The Tensolr tensor to wrap

**Example:**
```python
param = Parameter(Tensolr([1, 2, 3]))
```

#### `Linear`
Linear (fully connected) layer.

**Constructor:**
```python
Linear(in_features, out_features, bias=True)
```

**Arguments:**
- `in_features`: Size of each input sample
- `out_features`: Size of each output sample
- `bias`: Whether to include bias (default: True)

**Attributes:**
- `weight`: The learnable weights of the module
- `bias`: The learnable bias of the module

**Example:**
```python
linear = Linear(10, 5)  # 10 input features, 5 output features
input_tensor = Tensolr.randn((3, 10))  # batch size 3
output = linear(input_tensor)  # shape: (3, 5)
```

#### `ReLU`
ReLU activation function.

**Constructor:**
```python
ReLU()
```

**Example:**
```python
relu = ReLU()
input_tensor = Tensolr([[-1, 2], [3, -4]])
output = relu(input_tensor)  # [[0, 2], [3, 0]]
```

#### `Sigmoid`
Sigmoid activation function.

**Constructor:**
```python
Sigmoid()
```

**Example:**
```python
sigmoid = Sigmoid()
input_tensor = Tensolr([[-1, 0], [1, 2]])
output = sigmoid(input_tensor)  # Apply sigmoid function
```

#### `Tanh`
Tanh activation function.

**Constructor:**
```python
Tanh()
```

**Example:**
```python
tanh = Tanh()
input_tensor = Tensolr([[-1, 0], [1, 2]])
output = tanh(input_tensor)  # Apply tanh function
```

#### `Sequential`
A sequential container that stacks modules in the order they are passed.

**Constructor:**
```python
Sequential(*modules)
```

**Arguments:**
- `modules`: Variable length module list

**Example:**
```python
model = Sequential(
    Linear(784, 128),
    ReLU(),
    Linear(128, 10),
    Sigmoid()
)
```

#### `MSE`
Mean Squared Error loss function.

**Constructor:**
```python
MSE()
```

**Forward Method:**
```python
forward(y_pred, y_true)
```

**Arguments:**
- `y_pred`: Predicted values
- `y_true`: True values

**Example:**
```python
mse_loss = MSE()
predictions = Tensolr([[1.0, 2.0], [3.0, 4.0]])
targets = Tensolr([[1.5, 2.5], [2.5, 3.5]])
loss = mse_loss(predictions, targets)
```

---

### Optimizers

#### `Optimizer`
Base class for all optimizers.

**Methods:**
- `zero_grad()`: Clears gradients
- `step()`: Updates parameters, should be implemented in subclasses

#### `SGD`
Stochastic Gradient Descent optimizer.

**Constructor:**
```python
SGD(parameters, lr=0.01, momentum=0.0, weight_decay=0.0)
```

**Arguments:**
- `parameters`: Iterable of parameters to optimize
- `lr`: Learning rate (default: 0.01)
- `momentum`: Momentum factor (default: 0.0)
- `weight_decay`: Weight decay (L2 penalty) (default: 0.0)

**Example:**
```python
model = Linear(10, 1)
optimizer = SGD(model.parameters(), lr=0.01)
optimizer.zero_grad()
loss = compute_loss(model, data)  # Your loss computation
loss.backward()  # Compute gradients
optimizer.step()  # Update parameters
```

#### `Adam`
Adam optimizer.

**Constructor:**
```python
Adam(parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
```

**Arguments:**
- `parameters`: Iterable of parameters to optimize
- `lr`: Learning rate (default: 0.001)
- `betas`: Coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
- `eps`: Term added to the denominator to improve numerical stability (default: 1e-8)
- `weight_decay`: Weight decay (L2 penalty) (default: 0.0)

**Example:**
```python
model = Linear(10, 1)
optimizer = Adam(model.parameters(), lr=0.001)
optimizer.zero_grad()
loss = compute_loss(model, data)  # Your loss computation
loss.backward()  # Compute gradients
optimizer.step()  # Update parameters
```

#### `RMSprop`
RMSprop optimizer.

**Constructor:**
```python
RMSprop(parameters, lr=0.001, alpha=0.99, eps=1e-8, weight_decay=0.0)
```

**Arguments:**
- `parameters`: Iterable of parameters to optimize
- `lr`: Learning rate (default: 0.001)
- `alpha`: Smoothing constant (default: 0.99)
- `eps`: Term added to the denominator to improve numerical stability (default: 1e-8)
- `weight_decay`: Weight decay (L2 penalty) (default: 0.0)

**Example:**
```python
model = Linear(10, 1)
optimizer = RMSprop(model.parameters(), lr=0.001)
optimizer.zero_grad()
loss = compute_loss(model, data)  # Your loss computation
loss.backward()  # Compute gradients
optimizer.step()  # Update parameters
```

---

## Complete Example

Here's a complete example showing how to use Tensolr to build and train a simple neural network:

```python
from tensolr.api import nn, optim
import tensolr

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 10)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(10, 1)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# Create model, loss function and optimizer
model = SimpleNet()
criterion = nn.MSE()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Sample data
X = tensolr.Tensolr([[0, 0], [0, 1], [1, 0], [1, 1]])  # XOR inputs
y = tensolr.Tensolr([[0], [1], [1], [0]])              # XOR outputs

# Training loop
for epoch in range(1000):
    optimizer.zero_grad()
    
    # Forward pass
    predictions = model(X)
    
    # Compute loss
    loss = criterion(predictions, y)
    
    # Backward pass
    loss.backward()  # This would use the computational graph
    
    # Update weights
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.data}')
```