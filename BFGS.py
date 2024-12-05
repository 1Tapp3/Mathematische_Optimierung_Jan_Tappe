import numpy as np
from DifferentiableFunction import IDifferentiableFunction, DifferentiableFunction
from Set import AffineSpace, MultidimensionalInterval
import LineSearch


class BFGS(object):

    def __init__(self):
        super().__init__()

    def Minimize(self, function: IDifferentiableFunction, startingpoint: np.array, iterations: int = 100, tol_x=1e-5, tol_y=1e-5) -> np.array:
        x = startingpoint
        n = x.shape[0]
        H = np.identity(n)  # Approximate inverse Hessian
        y = function.evaluate(x)
        linesearch = LineSearch.LineSearch()
        gradient = function.jacobian(x).reshape([-1])

        if isinstance(function.domain, MultidimensionalInterval):
            lower_bounds = function.domain.lower_bounds
            upper_bounds = function.domain.upper_bounds
        else:
            lower_bounds = np.full(startingpoint.shape, -np.inf)
            upper_bounds = np.full(startingpoint.shape, np.inf)

        for step in range(iterations):
            if np.linalg.norm(gradient) == 0:
                return x

            # directional search
            p = - np.matmul(H, gradient)
            alpha = linesearch.LineSearchForWolfeConditions(
                function, startingpoint=x, direction=p, lower_bounds=lower_bounds, upper_bounds=upper_bounds)
            s = alpha*p
            x = x+s
            x = np.minimum(upper_bounds, np.maximum(lower_bounds, x))
            if np.linalg.norm(s) < tol_x:
                break

            # update inverse Hessian using BFGS, this needs the search drection to satisfy the strong Wolfe condition
            y_new = function.evaluate(x)
            delta_y = y - y_new
            if delta_y < tol_y:
                break
            gradient_old = gradient
            gradient = function.jacobian(x).reshape([-1])
            delta_grad = gradient - gradient_old
            scaling = np.dot(s, delta_grad)
            # scaling should alway be positive due to the curvature condition (second Wolfe condition)
            # if it is not positive, the new Hessian might not be positive definite
            # Hence, we skip the Hessian update in that case (which should not happen for decent line searches)
            if scaling > 0:
                H = H + (scaling + np.dot(delta_grad, np.matmul(H, delta_grad))) / \
                    scaling**2 * np.outer(s, s) - 1.0/scaling*(np.matmul(H,
                                                                         np.outer(delta_grad, s))+np.matmul(np.outer(s, delta_grad), H))
            y = y_new

        return x
