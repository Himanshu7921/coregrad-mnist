from coregrad import Scalar
from blocks import LinearLayer

class NeuralNetwork:
    """
    Implements a fully-connected feedforward neural architecture as a composition
    of stacked non-linear affine operators.

    The network models a parameterized function:

    f_θ : R^(d_in) → R^(d_out)

    defined recursively as:
    h_(l+1) = φ(W_l * h_l + b_l)

    where:
    - θ = {W_l, b_l} for l = 1 to L
    - φ(.) = tanh(.)

    This architecture acts as a universal function approximator capable of learning
    highly non-linear mappings through gradient-based optimization over differentiable
    scalar computational graphs
    """
    def __init__(self, in_features: int, hidden_dim: int, out_features: int, n_layers: int):
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_features = out_features
        self.n_layers = n_layers
        self.layers = []

        # Adding 1st layer
        self.layers.append(LinearLayer(in_features, hidden_dim))

        # Adding hidden dims
        for _ in range(n_layers - 2):
            self.layers.append(LinearLayer(hidden_dim, hidden_dim))
        
        # Adding Final Layer
        self.layers.append(LinearLayer(hidden_dim, out_features))
    
    def forward(self, x: Scalar):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def __call__(self, x: Scalar):
        return self.forward(x)
    
    def __repr__(self):
        return f"[LinearLayer({self.in_features}, {self.hidden_dim})] | [LinearLayer({self.hidden_dim}, {self.hidden_dim})] x {self.n_layers - 2} | [LinearLayer({self.hidden_dim}, {self.out_features})]"
    
    def parameters(self):
        self.nn_params = []
        for layer in self.layers:
            self.nn_params.extend(layer.parameters())
        return self.nn_params

