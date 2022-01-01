import numpy as np
from .modules import Layer, softmax

class Model(Module):
    def __init__(self, dims : List[int], acts : List[Module]):
        self.layers = []
        for i, dim in enumerate(dims[:-1]):
            self.layers += [Layer(dim, dims[i+1], acts[i])]
            
    def __repr__(self):
        return "Model(\n  " + '\n  '.join(map(str, self.layers)) + '\n)'
        
    def forward(self, X : np.array):
        h = self.layers[0](X)
        for layer in self.layers[1:]:
            h = layer(h)
        return h
    
    def calc_metrics(self, y, preds):
        probs = softmax(preds)
        loss = np.mean(-np.sum(y.T*np.log(probs+1e-9), axis=0)) 
        acc = np.mean(np.argmax(probs, axis=0) == np.argmax(y, axis=1))
        return loss, acc
    
    def fit(self, X: np.array, y : np.array, lr: Union[float, List[float]]):
        if isinstance(lr, float):
            lr = [lr]*len(self.layers)
        else:
            assert len(lr) == len(self.layers), "Please specify LR for each layer."
            
        vectors = [X.T]
        for layer in self.layers:
            vectors += [layer(vectors[-1])]
            
        probs = softmax(vectors[-1])
        loss, acc = self.calc_metrics(y, vectors[-1])
        
        grad = probs - y.T
        for i in range(len(self.layers) - 1, -1, -1):
            grad = self.layers[i].backward(vectors[i], grad, lr[i])
        
        return loss, acc