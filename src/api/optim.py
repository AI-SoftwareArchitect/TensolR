from ..tensor import Tensolr
import numpy as np

class Optimizer:
    """Base class for optimizers"""
    def __init__(self, parameters):
        self.parameters = list(parameters)  # Convert to list to ensure it's iterable

    def zero_grad(self):
        """Clear gradients"""
        for param in self.parameters:
            if hasattr(param, 'tensor') and hasattr(param.tensor, '_node'):
                param.tensor._node.grad = None
            elif hasattr(param, '_node'):
                param._node.grad = None

    def step(self):
        """Update parameters"""
        raise NotImplementedError

class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer"""
    def __init__(self, parameters, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        # Initialize velocity for momentum
        self.velocities = [Tensolr.zeros(param.tensor.shape if hasattr(param, 'tensor') else param.shape) 
                          for param in self.parameters]
        
    def step(self):
        for i, param in enumerate(self.parameters):
            # Get the tensor from Parameter object or use directly
            tensor = param.tensor if hasattr(param, 'tensor') else param
            
            # Handle case where gradient is None
            if tensor._node is None or tensor._node.grad is None:
                # If no gradient information, skip the update
                continue
            
            grad = tensor._node.grad
            
            # Apply weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * tensor.data
                tensor._node.grad = grad  # Update the node grad with weight decay
            
            # Apply momentum
            if self.momentum != 0:
                self.velocities[i].data = self.momentum * self.velocities[i].data - self.lr * grad
                tensor.data += self.velocities[i].data
            else:
                # Standard SGD update
                tensor.data -= self.lr * grad

class Adam(Optimizer):
    """Adam optimizer"""
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Initialize moment estimates for each parameter
        self.step_counts = [0] * len(self.parameters)
        self.m = [Tensolr.zeros(param.tensor.shape if hasattr(param, 'tensor') else param.shape) 
                 for param in self.parameters]
        self.v = [Tensolr.zeros(param.tensor.shape if hasattr(param, 'tensor') else param.shape) 
                 for param in self.parameters]
        
    def step(self):
        for i, param in enumerate(self.parameters):
            tensor = param.tensor if hasattr(param, 'tensor') else param
            
            # Handle case where gradient is None
            if tensor._node is None or tensor._node.grad is None:
                # If no gradient information, skip the update
                continue
                
            grad = tensor._node.grad
            
            # Apply weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * tensor.data
            
            # Update step count
            self.step_counts[i] += 1
            
            # Update biased first moment estimate
            self.m[i].data = self.beta1 * self.m[i].data + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[i].data = self.beta2 * self.v[i].data + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i].data / (1 - self.beta1 ** self.step_counts[i])
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i].data / (1 - self.beta2 ** self.step_counts[i])
            
            # Update parameters
            tensor.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

class RMSprop(Optimizer):
    """RMSprop optimizer"""
    def __init__(self, parameters, lr=0.001, alpha=0.99, eps=1e-8, weight_decay=0.0):
        super().__init__(parameters)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Initialize moving average of squared gradients for each parameter
        self.mean_squared_grad = [Tensolr.zeros(param.tensor.shape if hasattr(param, 'tensor') else param.shape) 
                                 for param in self.parameters]
        
    def step(self):
        for i, param in enumerate(self.parameters):
            tensor = param.tensor if hasattr(param, 'tensor') else param
            
            # Handle case where gradient is None
            if tensor._node is None or tensor._node.grad is None:
                # If no gradient information, skip the update
                continue
                
            grad = tensor._node.grad
            
            # Apply weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * tensor.data
            
            # Update moving average of squared gradients
            self.mean_squared_grad[i].data = (self.alpha * self.mean_squared_grad[i].data + 
                                            (1 - self.alpha) * grad ** 2)
            
            # Update parameters
            tensor.data -= self.lr * grad / (np.sqrt(self.mean_squared_grad[i].data) + self.eps)