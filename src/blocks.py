import random
from coregrad import Scalar  # my own autograd engine: Available here: https://github.com/Himanshu7921/coregrad

class Neuron:
    """
    Implements a single non-linear computational unit parameterized by
    w ∈ R^d and b ∈ R.

    Given an input vector x, the neuron computes:

    y = tanh(w[i] * x[i] + b)

    This module forms the fundamental differentiable primitive of the network,
    where gradients are propagated through scalar-valued computational graphs
    constructed by the autograd engine.
    """
    def __init__(self, in_features: int):
        self.in_features = in_features
        self.weight = [Scalar(random.uniform(-1, 1)) for _ in range(in_features)]
        self.bias = Scalar(random.uniform(-1, 1))
    
    def __call__(self, x: Scalar):
        return self.forward(x)
    
    def forward(self, x: Scalar):
        y = []

        for xi, wi in zip(x, self.weight):
            y.append(xi * wi)

        act = sum(y, Scalar(0)) + self.bias

        tanh_act = act.tanh()
        return tanh_act
    
    def __repr__(self):
        return f"Neuron(in_features = {self.in_features})"
    
    def parameters(self):
        return self.weight + [self.bias]


class LinearLayer:
    """
    Implements a dense affine transformation followed by element-wise non-linearity
    through a collection of independent neurons.

    For an input matrix X ∈ R^(m × d), the layer approximates:

    H = φ(XW + b)

    where:
    - W ∈ R^(d × k)
    - b ∈ R^k
    - φ(.) = tanh(.)

    The layer serves as a learnable feature transformation operator enabling
    hierarchical representation learning in deep architectures
    """
    def __init__(self, in_features: int, out_features: int):
        self.neurons = []
        self.in_features = in_features
        self.out_features = out_features
        for _ in range(out_features):
            self.neurons.append(Neuron(in_features))
    
    def forward(self, x: Scalar):
        layer_output = []
        for l in self.neurons: # [n1(10), n2(10)]
            ni = l(x)
            layer_output.append(ni)
        return layer_output if len(layer_output) > 1 else layer_output[0]
    
    def __call__(self, x: Scalar):
        return self.forward(x)
    
    def __repr__(self):
        return f"LinearLayer({self.in_features}, {self.out_features})"
    
    def parameters(self):
        self.param = []
        for n in self.neurons:
            self.param.extend(n.parameters())
        return self.param