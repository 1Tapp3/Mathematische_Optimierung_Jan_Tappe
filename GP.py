import numpy as np
import math
from DifferentiableFunction import DifferentiableFunction
from Set import AffineSpace


class GP(object):
    def __init__(self, data_x: np.array, data_y: np.array):
        super().__init__()
        assert data_x.shape[0] == data_y.shape[0], "need as many x-positions as y-positions."
        self.data_x = data_x
        self.data_y = data_y
        self.n = self.data_x.shape[0]
        self.d = self.data_x.shape[1]
        self.kernel = lambda x1, x2: math.exp(-0.5*np.linalg.norm(x1-x2)**2)

    def __K(self) -> np.array:
        """The covariance matrix of this GP"""
        if not hasattr(self, 'K'):
            self.K = np.array([[self.kernel(self.data_x[i, :], self.data_x[j, :]) + (i == j) * 1e-5
                                for i in range(self.n)] for j in range(self.n)]).reshape(self.n, self.n)
        return self.K

    def __L(self) -> np.array:
        """The Cholesky factor of the covariance matrix K (notation as in Rasmussen&Williams) of this GP"""
        if not hasattr(self, 'L'):
            self.L = np.linalg.cholesky(self.__K())
        return self.L

    def __alpha(self) -> np.array:
        """The vector alpha (notation as in Rasmussen&Williams) of this GP"""
        if not hasattr(self, 'alpha'):
            # this is numerically less stable than using the cholesky decomposition
            self.alpha = np.linalg.solve(self.__K(), self.data_y)
        return self.alpha

    def __ks(self, x: np.array) -> np.array:
        """The vector k_*=k(x_*,X) (notation as in Rasmussen&Williams) given of this GP"""
        return np.array([self.kernel(x, self.data_x[i, :]) for i in range(self.n)])

    def __dks(self, x: np.array) -> np.array:
        """The derivative of the vector k_*=k(x_*,X) (notation as in Rasmussen&Williams) given of this GP"""
        return np.array([list(self.kernel(x, self.data_x[i, :])*(self.data_x[i, :]-x)) for i in range(self.n)])

    def PosteriorMean(self):
        return DifferentiableFunction(
            name="GP_posterior_mean",
            domain=AffineSpace(self.d),
            evaluate=lambda x: np.dot(self.__alpha(), self.__ks(x)),
            jacobian=lambda x: np.dot(self.__alpha(), self.__dks(x))
        )

    def PosteriorVariance(self):
        return DifferentiableFunction(
            name="GP_posterior_variance",
            domain=AffineSpace(self.d),
            evaluate=lambda x: np.array([self.kernel(
                x, x)-np.linalg.norm(np.linalg.solve(self.__L(), self.__ks(x)))**2]),
            jacobian=lambda x: 0-2 *
            np.reshape(np.dot(np.linalg.solve(self.__L(), self.__ks(x)),
                              np.linalg.solve(self.__L(), self.__dks(x))), (1, -1))
        )

    def PosteriorStandardDeviation(self):
        sqrt = DifferentiableFunction(name="sqrt", domain=AffineSpace(
            1), evaluate=lambda x: np.sqrt(x), jacobian=lambda x: np.reshape(0.5/np.sqrt(x), (1, 1)))
        return DifferentiableFunction.FromComposition(sqrt, self.PosteriorVariance())
