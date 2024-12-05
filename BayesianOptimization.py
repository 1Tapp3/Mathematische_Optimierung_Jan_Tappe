import numpy as np
import math
from typing import Callable
from DifferentiableFunction import IDifferentiableFunction
from Set import AffineSpace
from GP import GP
from SQP import SQP


class BO(object):
    def __init__(self):
        super().__init__()

    def Minimize(self, function: IDifferentiableFunction, iterations: int = 50) -> np.array:

        domain = function.domain
        d = domain._ambient_dimension
        data_x = np.empty((0, d))
        data_y = np.empty((0,))
        gp = GP(data_x=data_x, data_y=data_y)
        sqp = SQP()

        for step in range(iterations):
            # UCB with 2 sigma, added with 0 times the original function to get the domain
            acquisition_function = gp.PosteriorMean(
            )-2*gp.PosteriorStandardDeviation() + 0*function

            # new measurement point
            startingpoint = domain.point()
            x = sqp.Minimize(acquisition_function,
                             startingpoint=startingpoint)
            # measure
            y = function.evaluate(x)

            # update data
            data_x = np.concatenate((data_x, x.reshape(1, -1)), axis=0)
            data_y = np.concatenate((data_y, y), axis=0)
            gp = GP(data_x=data_x, data_y=data_y)

        return data_x[np.argmin(data_y), :]
