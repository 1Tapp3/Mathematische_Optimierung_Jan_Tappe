import numpy as np
from DifferentiableFunction import IDifferentiableFunction, DifferentiableFunction
from Set import AffineSpace, MultidimensionalInterval
import LineSearch


class BFGS(object):

    def __init__(self):
        super().__init__()

    def Minimize(self, function: IDifferentiableFunction, startingpoint: np.array, iterations: int = 100, tol_x=1e-5, tol_y=1e-5) -> np.array:
        """Optimized BFGS method using the Sherman-Morrison formula for efficiency."""
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

            # Direction search: avoid full matrix-vector multiplication
            p = -H @ gradient  
            alpha = linesearch.LineSearchForWolfeConditions(
                function, startingpoint=x, direction=p, lower_bounds=lower_bounds, upper_bounds=upper_bounds
            )
            s = alpha*p
            x = np.minimum(upper_bounds, np.maximum(lower_bounds, x + s))

            if np.linalg.norm(s) < tol_x:
                break

            # Function value and gradient update
            y_new = function.evaluate(x)
            delta_y = y - y_new
            if delta_y < tol_y:
                break
            gradient_old = gradient
            gradient = function.jacobian(x).reshape([-1])
            delta_grad = gradient - gradient_old
            scaling = np.dot(s, delta_grad)

            if scaling > 0:
                # **Optimized Hessian Update using the Sherman-Morrison formula**
                rho = 1.0 / scaling
                V = np.eye(n) - rho * np.outer(s, delta_grad)
                H = V @ H @ V.T + rho * np.outer(s, s)

            y = y_new

        return x

