import numpy as np
from DifferentiableFunction import IDifferentiableFunction, DifferentiableFunction
from Set import AffineSpace


class GradientDescent(object):

    def __init__(self):
        super().__init__()

    def Minimize(self, function: IDifferentiableFunction, startingpoint: np.array, iterations: int = 100, learningrate: float = 0.1) -> np.array:
        x = startingpoint
        for step in range(iterations):
            gradient = function.jacobian(x).reshape([-1])
            x = x - learningrate * gradient
        return x
