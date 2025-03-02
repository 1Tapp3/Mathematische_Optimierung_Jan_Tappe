import numpy as np
import math
from math import gamma
from abc import ABC, abstractmethod
from DifferentiableFunction import DifferentiableFunction
from Set import AffineSpace

#I added an Abstract Class for Kenrels and implemented MaternKernel() and PeriodicKernel()

class Kernel(ABC):
    @abstractmethod
    def compute(self, x1: np.ndarray, x2: np.ndarray) -> float:
        pass
    
    @abstractmethod
    def derivative(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        pass

class RBFKernel(Kernel):
    def __init__(self, lengthscale=1.0):
        self.lengthscale = lengthscale
    
    def compute(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return math.exp(-0.5 * (np.linalg.norm(x1 - x2) / self.lengthscale) ** 2)
    
    def derivative(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        k = self.compute(x1, x2)
        return k * (x2 - x1) / (self.lengthscale ** 2)

import numpy as np
import math

class MaternKernel:
    def __init__(self, lengthscale=1.0, nu=2.5):
        self.lengthscale = lengthscale
        self.nu = nu

    def modified_bessel_second_kind(self, nu: float, x: float) -> float:
        if x == 0:
            return float('inf')
        elif x > 5:
            return math.sqrt(math.pi / (2 * x)) * math.exp(-x)
        else:
            summe = 0.0
            for k in range(10):
                summe += ((-1) ** k / math.factorial(k)) * ((x / 2) ** (2 * k))
            return (math.pi / (2 * math.sin(math.pi * nu))) * (summe / (x ** nu))

    def compute(self, x1: np.ndarray, x2: np.ndarray) -> float:
        r = np.linalg.norm(x1 - x2) / self.lengthscale
        if r == 0:
            return 1.0
        
        factor = (2 ** (1 - self.nu)) / math.gamma(self.nu)
        scaled_r = np.sqrt(2 * self.nu) * r
        return factor * (scaled_r ** self.nu) * self.modified_bessel_second_kind(self.nu, scaled_r)

    def derivative(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        r = np.linalg.norm(x1 - x2) / self.lengthscale
        if r == 0:
            return np.zeros_like(x1)
        
        k = self.compute(x1, x2)
        scaled_r = np.sqrt(2 * self.nu) * r
        bessel_ratio = self.modified_bessel_second_kind(self.nu - 1, scaled_r) / self.modified_bessel_second_kind(self.nu, scaled_r)
        
        factor = (np.sqrt(2 * self.nu) / self.lengthscale) * bessel_ratio
        return k * factor * (x2 - x1) / r

    
class PeriodicKernel(Kernel):
    def __init__(self, lengthscale=1.0, period=1.0):
        self.lengthscale = lengthscale
        self.period = period
    
    def compute(self, x1: np.ndarray, x2: np.ndarray) -> float:
        r = np.linalg.norm(x1 - x2)
        return math.exp(-2 * (math.sin(math.pi * r / self.period) ** 2) / (self.lengthscale ** 2))
    
    def derivative(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        r = np.linalg.norm(x1 - x2)
        if r == 0:
            return np.zeros_like(x1)
        
        k = self.compute(x1, x2)
        factor = (-4 * math.pi / (self.lengthscale ** 2 * self.period)) * math.sin(2 * math.pi * r / self.period)
        
        return factor * k * (x1 - x2) / r

class GP:
    def __init__(self, data_x: np.array, data_y: np.array, kernel: Kernel = None,):
        self.data_x = data_x
        self.data_y = data_y
        self.n = self.data_x.shape[0]
        self.d = self.data_x.shape[1]
        self.kernel = kernel if kernel is not None else RBFKernel()

    def __K(self) -> np.array:
        """The covariance matrix of this GP"""
        return np.array([[self.kernel.compute(self.data_x[i, :], self.data_x[j, :]) + (i == j) * 1e-5
                          for i in range(self.n)] for j in range(self.n)]).reshape(self.n, self.n)

    def __L(self) -> np.array:
        """The Cholesky factor of the covariance matrix K (notation as in Rasmussen&Williams) of this GP"""
        return np.linalg.cholesky(self.__K())

    def __alpha(self) -> np.array:
        """The vector alpha (notation as in Rasmussen&Williams) of this GP"""
        return np.linalg.solve(self.__K(), self.data_y)

    def __ks(self, x: np.array) -> np.array:
        """The vector k_*=k(x_*,X) (notation as in Rasmussen&Williams) given of this GP"""
        return np.array([self.kernel.compute(x, self.data_x[i, :]) for i in range(self.n)])

    def __dks(self, x: np.array) -> np.array:
        return np.array([self.kernel.derivative(x, self.data_x[i, :]) for i in range(self.n)])

    def PosteriorMean(self) -> DifferentiableFunction:
        return DifferentiableFunction(
            name="GP_posterior_mean",
            domain=AffineSpace(self.d),
            evaluate=lambda x: np.dot(self.__alpha(), self.__ks(x)),
            jacobian=lambda x: np.dot(self.__alpha(), self.__dks(x))
        )

    def PosteriorVariance(self) -> DifferentiableFunction:
        return DifferentiableFunction(
            name="GP_posterior_variance",
            domain=AffineSpace(self.d),
            evaluate=lambda x: np.array([self.kernel.compute(
                x, x)-np.linalg.norm(np.linalg.solve(self.__L(), self.__ks(x)))**2]),
            jacobian=lambda x: -2 *
            np.reshape(np.dot(np.linalg.solve(self.__L(), self.__ks(x)),
                              np.linalg.solve(self.__L(), self.__dks(x))), (1, -1))
        )

    def PosteriorStandardDeviation(self) -> DifferentiableFunction:
        return DifferentiableFunction(
            name="GP_posterior_stddev",
            domain=AffineSpace(self.d),
            evaluate=lambda x: np.sqrt(self.PosteriorVariance().evaluate(x)),
            jacobian=lambda x: (0.5 / np.sqrt(self.PosteriorVariance().evaluate(x))) * self.PosteriorVariance().jacobian(x)
        )
