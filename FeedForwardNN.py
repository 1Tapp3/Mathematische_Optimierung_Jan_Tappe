import numpy as np
import math
from DifferentiableFunction import IDifferentiableFunction, DifferentiableFunction
from Set import AffineSpace
from GradientDescent import GradientDescent


class ReLUFeedForwardNN(object):
    """This class is a feed forward neural network with ReLU activation functions"""
    # This class is very much for instruction only and not efficient (not even close)

    def __init__(self):
        super().__init__()

        self.dims = [2, 10, 1]
        self.input_dim = self.dims[0]
        self.hidden_dim = self.dims[1]
        self.ReLU = DifferentiableFunction.ReLU(
            dimension=self.hidden_dim)

        # He initialization of the parameters, stored as vectors (will be reshaped to matrices later)
        self.lins = [np.random.normal(0, math.sqrt(2.0/self.dims[i]), self.dims[i+1] * self.dims[i])
                     for i in range(len(self.dims)-1)]
        self.bias = np.random.normal(0.0, 0.1, self.hidden_dim)
        self.params = np.concatenate(
            (self.lins[0], self.bias, self.lins[1]))

        self.lin1_range = range(self.input_dim*self.hidden_dim)
        self.bias_range = range(
            self.input_dim*self.hidden_dim, (self.input_dim+1)*self.hidden_dim)
        self.lin2_range = range(
            (self.input_dim+1)*self.hidden_dim, len(self.params))

    def ToFunction(self) -> IDifferentiableFunction:
        """Returns the differentiable (modulo ReLU) function that evaluates the network from inputs (having fixed the parameters)"""
        # reshape parameters into matrices
        A0 = np.reshape(self.params[self.lin1_range],
                        (self.hidden_dim, self.input_dim))
        f0 = DifferentiableFunction.LinearMapFromMatrix(
            A0)
        # f0 = DifferentiableFunction.Debug(f0)
        v = self.params[self.bias_range]
        bias = DifferentiableFunction.TranslationByVector(
            v)
        # bias = DifferentiableFunction.Debug(bias)
        A1 = np.reshape(self.params[self.lin2_range], (1, self.hidden_dim))
        f1 = DifferentiableFunction.LinearMapFromMatrix(
            A1)
        # f1 = DifferentiableFunction.Debug(f1)
        compose = DifferentiableFunction.FromComposition
        return compose(compose(compose(f1, self.ReLU), bias), f0)

    def ToLoss(self, data_x: np.array, data_y: np.array) -> IDifferentiableFunction:
        """Returns the differentiable (modulo ReLU) function that evaluates the network on parameters for evaluating the NN on fixed given data"""
        # Obviously, this is not made for batch training or many other interesting NN techniques
        # Obviously, there are manny additional problems like using a loop over the data
        # Obviously, taking the Kronecker product is highly problematic in terms of space complex (and indirectly time) complexity
        assert data_x.shape[0] == data_y.shape[0], "need as many labels as data points"

        compose = DifferentiableFunction.FromComposition

        # This is the zero function
        loss = 0

        for i in range(data_x.shape[0]):
            datum = data_x[i, :]

            # A*x = (I\otimes x)*vec(A), correct, but slow
            mat = np.kron(np.eye(self.hidden_dim),
                          datum.reshape((1, -1)))
            lin1 = DifferentiableFunction.LinearMapFromMatrix(
                mat)
            id1 = DifferentiableFunction.Identity(
                AffineSpace(self.hidden_dim * 2))
            lin1 = lin1.CartesianProduct(id1)

            mat_bias = np.concatenate(
                (np.eye(self.hidden_dim), np.eye(self.hidden_dim)), axis=1)
            bias = DifferentiableFunction.LinearMapFromMatrix(
                mat_bias)
            id2 = DifferentiableFunction.Identity(
                AffineSpace(self.hidden_dim))
            bias = bias.CartesianProduct(id2)

            relu = self.ReLU.CartesianProduct(id2)

            inner_product_from_cartesian_product = DifferentiableFunction(
                name="inner_product_from_cartesian_product",
                domain=AffineSpace(
                    2*(len(self.params)-self.hidden_dim * self.input_dim)),
                evaluate=lambda x: np.dot(x[:len(x)//2], x[len(x)//2:]),
                jacobian=lambda x: np.concatenate(
                    (x[x.shape[0]//2:], x[:x.shape[0]//2]), axis=0).reshape(1, -1)
            )

            local_loss = compose(compose(compose(
                inner_product_from_cartesian_product, relu), bias), lin1)
            local_loss = (local_loss - data_y[i])**2
            loss = local_loss + loss

        # mean (of mean squared error)
        loss = (1.0 / data_x.shape[0]) * loss

        return loss
