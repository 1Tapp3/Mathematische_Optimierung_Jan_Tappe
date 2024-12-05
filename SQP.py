import numpy as np
from QP import QP
from Set import AffineSpace, MultidimensionalInterval
from SetsFromFunctions import BoundedSet
import LineSearch
from DifferentiableFunction import DifferentiableFunction, IDifferentiableFunction


class SQP(object):
    def __init__(self):
        super().__init__()

    def Minimize(self, function: IDifferentiableFunction, startingpoint: np.array, iterations: int = 100, tol_x=1e-5, tol_y=1e-5):
        """Minimize the function using SQP"""

        x = startingpoint
        n = x.shape[0]
        y = function.evaluate(x)
        gradient = function.jacobian(x).reshape([-1])

        domain = function.domain
        # of course, SQP is possible without this condition, but much more complicated to implement.
        assert domain.contains(x), "starting point must be valid"

        lowerBounds = np.array([float('-inf')]*n)
        upperBounds = np.array([float('inf')]*n)
        if isinstance(domain, MultidimensionalInterval):
            lowerBounds = domain.lower_bounds
            upperBounds = domain.upper_bounds

        ineq = DifferentiableFunction(
            name="zero", domain=AffineSpace(n), evaluate=lambda x: np.array([0]), jacobian=lambda x: np.array([[0]]))
        if isinstance(domain, BoundedSet):
            ineq = domain.InequalityConstraints

        # Subroutine solvers
        qp = QP()
        linesearch = LineSearch.LineSearch()

        H = np.identity(n)  # Approximate Hessian
        for i in range(iterations):

            # linearize problem and solve QP for search direction
            ineq_eval = ineq.evaluate(x)
            ineq_jacobian = ineq.jacobian(x)
            p = qp.QP(H, gradient.transpose(), ineq_jacobian, -ineq_eval)

            alpha = linesearch.LineSearchForFeasibility(
                function, startingpoint=x, direction=p, alpha=1.0)

            # Change current position
            s = alpha*p
            x = x+s
            assert domain.contains(x), "all intermediate points must be valid"

            # consider bounds -> This should happen in line search and in the QP
            x = np.maximum(np.minimum(x, upperBounds), lowerBounds)

            # Update Hessian using BFGS, this needs the search drection to satisfy the strong Wolfe condition
            y_new = function.evaluate(x)
            delta_y = y - y_new
            if abs(delta_y) < tol_y and domain.contains(x):
                #     print("y " + str(delta_y) + " < tol " + str(tol_y))
                break
            gradient_old = gradient
            gradient = function.jacobian(x).reshape([-1])
            delta_grad = gradient - gradient_old
            scaling = np.dot(s, delta_grad)
            # scaling should alway be positive due to the curvature condition (second Wolfe condition)
            # if it is not positive, the new Hessian might not be positive definite
            # Hence, we skip the Hessian update in that case (which should not happen for decent line searches)
            if scaling > 0:
                H = H + 1.0/scaling*np.outer(delta_grad, delta_grad)-1.0/(
                    np.dot(s, np.matmul(H, s)))*np.matmul(np.matmul(H, np.outer(s, s)), H)
                assert np.sum(np.abs((H-np.transpose(H)))) / np.sum(np.abs(H)
                                                                    ) < 1e-4, "Hessian approximation needs to be symmetric"
            y = y_new

            # todo Buch p. 536f

        return x
