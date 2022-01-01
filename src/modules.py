import numpy as np

class Module:
    def forward(self, X):
        raise NotImplementedError
        
    def __call__(self, X):
        return self.forward(X)
    
    def backward(self, X, grad):
        raise NotImplementedError


class Sigmoid(Module):
    def forward(self, X):
        return 1/(1 + np.exp(-X))
    
    def backward(self, X, grad):
        z = self.forward(X)
        return grad*z*(1 - z)
    def __repr__(self):
        return 'Sigmoid'
    
    
class Linear(Module):
    def forward(self, X):
        return X
    def backward(self, X, grad):
        return grad
    def __repr__(self):
        return 'Linear'
    
    
class ReLU(Module):
    def forward(self, X):
        return np.maximum(X, 0)
    def backward(self, X, grad): 
        grad_t = np.array(grad, copy = True)
        grad_t[X <= 0] = 0
        return grad_t
    def __repr__(self):
        return 'ReLU'
        
    
class TanH(Module):
    def forward(self, X):
        return np.tanh(X)
    
    def backward(self, X, grad):
        z = self.forward(X)
        return grad* (1 - z**2)
    def __repr__(self):
        return 'TanH'
        
def softmax(x, axis=0):
    s = np.exp(x - x.max(axis=axis))
    return s/s.sum(axis=axis)


class Layer(Module):
    def __init__(self, dim_in : int, dim_out : int, act : Module):
        self.W = np.random.randn(dim_out, dim_in)
        self.B = np.zeros(shape=(dim_out,1), dtype=float)
        self.act = act
        
    def __str__(self):
        return f'Layer(W={self.W.shape}, B={self.B.shape}, act={self.act})'
    
    def __repr__(self):
        return self.__str__()
    
    def forward(self, X):
        o = self.act(self.W@X + self.B)
        return o
    
    def backward(self, X : np.array, grad: np.array, lr : float):
        h = self.forward(X)
        m = h.shape[1]
        grad = self.act.backward(h, grad)
#         print('grad norm:' ,np.linalg.norm(grad))
        grad_W = (1/m) * grad @ X.T
        grad_B = (1/m) * np.sum(grad, axis=1, keepdims=True)
        grad = self.W.T @ grad
#         print('gradW:', grad_W.shape, 'gradB:', grad_B.shape, 'grad act:', grad.shape)
        
        self.W -= lr*grad_W
        self.B -= lr*grad_B
        return grad