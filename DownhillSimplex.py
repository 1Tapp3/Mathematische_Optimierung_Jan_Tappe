import numpy as np
from typing import Callable, Tuple


class DownhillSimplex:
    def __init__(self, func: Callable[[np.ndarray], float], x0: np.ndarray):
        """
        Initialize the Downhill Simplex optimizer.

        Parameters:
        - func: The objective function to minimize. Should take a numpy array as input and return a float.
        - x0: Initial guess for the minimum as a numpy array.
        """
        self.func = func
        self.x0 = np.array(x0, dtype=float)
        self.lower_bounds = np.full(x0.shape, -np.inf)
        self.upper_bounds = np.full(x0.shape, np.inf)

    def set_bounds(self, bounds: Tuple[np.ndarray, np.ndarray]):
        """
        Set bounds for the variables.

        Parameters:
        - bounds: A tuple of two numpy arrays (lower_bounds, upper_bounds).
        """
        self.lower_bounds, self.upper_bounds = bounds

    def _apply_bounds(self, point: np.ndarray) -> np.ndarray:
        """Apply the set bounds to the point, if bounds are defined."""
        return np.minimum(self.upper_bounds, np.maximum(self.lower_bounds, point))

    def optimize(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 2.0,
        sigma: float = 0.5,
        max_iter: int = 2000,
        tol: float = 1e-8
    ) -> Tuple[np.ndarray, float]:
        """
        Perform the Downhill Simplex optimization.

        Parameters:
        - alpha: Reflection coefficient (default 1.0).
        - beta: Contraction coefficient (default 0.5).
        - gamma: Expansion coefficient (default 2.0).
        - sigma: Shrinkage coefficient (default 0.5).
        - max_iter: Maximum number of iterations (default 2000).
        - tol: Tolerance for convergence (default 1e-8).

        Returns:
        - best_point: The best point found during optimization.
        - best_value: The value of the objective function at the best point.
        """
        dim = len(self.x0)

        # Initialize the simplex
        simplex = np.empty((dim + 1, dim), dtype=float)
        f_values = np.empty(dim + 1, dtype=float)
        simplex[0] = self._apply_bounds(self.x0)
        f_values[0] = self.func(simplex[0])

        for i in range(dim):
            x = np.copy(self.x0)
            x[i] += 0.5 if x[i] == 0 else 0.2 * abs(x[i])
            simplex[i + 1] = self._apply_bounds(x)
            f_values[i + 1] = self.func(simplex[i + 1])

        # Main optimization loop
        for iteration in range(max_iter):
            # Order the simplex points
            worst = np.argmax(f_values)
            best = np.argmin(f_values)
            second_worst = np.argsort(f_values)[-2]

            # Compute the centroid of all points except the worst
            centroid = np.mean(simplex[np.arange(dim + 1) != worst], axis=0)

            # Reflection
            reflected = self._apply_bounds(centroid + alpha * (centroid - simplex[worst]))
            f_reflected = self.func(reflected)

            if f_values[best] <= f_reflected < f_values[second_worst]:
                simplex[worst], f_values[worst] = reflected, f_reflected
            elif f_reflected < f_values[best]:
                # Expansion
                expanded = self._apply_bounds(centroid + gamma * (reflected - centroid))
                f_expanded = self.func(expanded)
                if f_expanded < f_reflected:
                    simplex[worst], f_values[worst] = expanded, f_expanded
                else:
                    simplex[worst], f_values[worst] = reflected, f_reflected
            else:
                # Contraction
                contracted = self._apply_bounds(centroid + beta * (simplex[worst] - centroid))
                f_contracted = self.func(contracted)
                if f_contracted < f_values[worst]:
                    simplex[worst], f_values[worst] = contracted, f_contracted
                else:
                    # Shrink the simplex
                    for i in range(1, dim + 1):
                        simplex[i] = self._apply_bounds(simplex[best] + sigma * (simplex[i] - simplex[best]))
                        f_values[i] = self.func(simplex[i])

            # Check for convergence
            if np.max(np.abs(f_values - f_values[best])) < tol:
                break

        return simplex[best], f_values[best]


# Define the Rosenbrock function (banana function)
def rosenbrock(x: np.ndarray, a: float = 1, b: float = 100) -> float:
    return (a - x[0])**2 + b * (x[1] - x[0]**2)**2


# Example usage with bounds
optimizer = DownhillSimplex(func=rosenbrock, x0=np.array([5, 3]))
optimizer.set_bounds((np.array([-6, -6]), np.array([6, 6])))
best_point, best_value = optimizer.optimize()
print("Best point:", best_point)
print("Best value:", best_value)
