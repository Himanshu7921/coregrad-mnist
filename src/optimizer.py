import numpy as np
from typing import List
from config import config
from coregrad import Scalar

class Adam:
    """
    Implements Adan Optimizer mentioned in the paper "ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION"
    Paper link: https://arxiv.org/pdf/1412.6980
    """
    def __init__(self,
                 parameters: List[Scalar],
                 lr: float = config["lr"],
                 beta1: float = config["beta1"],
                 beta2: float = config["beta2"],
                 eps: float = 1e-8):
        
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = [0.0 for _ in self.parameters]
        self.v = [0.0 for _ in self.parameters]

        self.t = 0

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0
    
    def step(self):
        """
        performs one step of Adam Optimizer
        """
        self.t +=1

        # current gradient at timestep = t
        for i, p in enumerate(self.parameters):
            g_t = p.grad # Get gradients w.r.t. stochastic objective at timestep t
            self.m[i] = (self.beta1 * self.m[i] + (1 - self.beta1) * g_t) # Update biased first moment estimate
            self.v[i] = (self.beta2 * self.v[i] + (1 - self.beta2) * (g_t**2)) # Update biased second raw moment estimate

            m_hat = self.m[i] / (1 - (self.beta1 ** self.t)) # Compute bias-corrected first moment estimate
            v_hat = self.v[i] / (1 - (self.beta2 ** self.t)) # Compute bias-corrected second raw moment estimate

            # update parameters
            p.data -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps))

