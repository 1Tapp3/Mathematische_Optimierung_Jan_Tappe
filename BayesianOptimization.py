import numpy as np
import math
from typing import Callable
from DifferentiableFunction import IDifferentiableFunction
from Set import AffineSpace
from GP import *
from SQP import SQP


class BO(object):
    def __init__(self, data_x: np.array = None, data_y: np.array = None, kernel: Kernel = None):
        """Initializes Bayesian Optimization with optional prior data and kernel settings.
        
        Args:
            data_x (np.array, optional): Input data points.
            data_y (np.array, optional): Observed function values.
            Kernel select kernelfunction to use with BO
            nu (float, optional): Smoothness parameter for the Matérn kernel. Defaults to 2.5.
            lengthscale (float, optional): Lengthscale parameter controlling kernel behavior. Defaults to 1.0.
        """
        self.data_x = data_x if data_x is not None else np.empty((0, 0))
        self.data_y = data_y if data_y is not None else np.empty((0,))
        self.kernel = kernel if kernel is not None else RBFKernel()
        assert self.data_x.shape[0] == self.data_y.shape[0]

    def Minimize(self, function: IDifferentiableFunction, iterations: int = 20) -> np.array:
        """Minimizes the given function using Bayesian Optimization with a selected kernel.
        
        Args:
            function (IDifferentiableFunction): The function to minimize.
            iterations (int, optional): Number of iterations. Defaults to 50.
        
        Returns:
            np.array: The optimized input point.
        """
        domain = function.domain
        d = domain._ambient_dimension
        if self.data_x.shape[0] == 0:
            data_x = np.empty((0, d))   
            data_y = np.empty((0,))
        else:
            data_x = np.copy(self.data_x)
            data_y = np.copy(self.data_y)    

        gp = GP(data_x=data_x, data_y=data_y, kernel=self.kernel)
        sqp = SQP()
        print("NEW")
        print(data_x)
        print(data_y)
        for step in range(iterations):
            acquisition_function = gp.PosteriorMean() - 2 * gp.PosteriorStandardDeviation() + 0 * function

            # new measurement point
            startingpoint = domain.point()
            x = sqp.Minimize(acquisition_function, startingpoint=startingpoint)
            
            # measure function value
            y = function.evaluate(x)

            # update data
            data_x = np.concatenate((data_x, x.reshape(1, -1)), axis=0)
            data_y = np.concatenate((data_y, y), axis=0)
            gp = GP(data_x=data_x, data_y=data_y, kernel=self.kernel)
            print("X")
            print(data_x)
            print("Y")
            print(data_y)
        return data_x[np.argmin(data_y), :]
