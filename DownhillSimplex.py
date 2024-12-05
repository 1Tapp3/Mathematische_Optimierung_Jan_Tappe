import numpy as np
from typing import Callable, Tuple

class DownhillSimplex:
    def __init__(self, func: Callable[[np.array], float], x0:np.array):
        self.func = func
        self.x0 = np.array(x0, dtype=float)
        self.lower_bounds = np.full(x0.shape, -np.inf)
        self.upper_Bounds = np.full(x0.shape, np.inf)

    def set_bounds(self, bounds: Tuple[np.array, np.array]):
        self.lower_bounds, self.upper_bounds = bounds

    def optimize(self, alpha: float = 1.0, beta: float = 0.5, gamma: float = 2.0, sigma: float = 0.5, max_iter: int = 2000, tol: float = 1e-8) -> np.array:
        """

        """
        def _apply_bounds(self, point: np.array) -> np.array:
            """Apply the set bounds to the point, if bounds are defined."""
            point = np.minimum(self.upper_bounds, np.maximum(self.lower_bounds, x))
            return point
        
        dim = len(self.x0)

        simplex = np.empty((dim + 1, dim), dtype=float)
        f_values = np.empty(dim + 1, dtype=float)
        simplex[0] = self.x0
        f_values[0] = self.func(self.x0)
        
        for i in range(dim):
            x = np.copy(self.x0)
            x[i] += 0.5 if x[i] == 0 else 0.20 * x[i]
            simplex[i+1] = _apply_bounds(x)
            f_values[i+1] = self.func(x)

        for iteration in range(max_iter):
            worst = np.argmax(f_values)
            best = np.argmin(f_values)
            second_worst = np.argpartition(f_values, -2)[-2]

            centroid = np.mean(simplex[np.arange(dim + 1) != worst], axis=0)
            reflected = _apply_bounds(centroid + alpha * (centroid - simplex[worst]))
            f_reflected = self.func(reflected)

            if f_values[best] <= f_reflected < f_values[second_worst]:
                simplex[worst], f_values[worst] = reflected, f_reflected

            elif f_reflected < f_values[best]:
                expanded = _apply_bounds(centroid + gamma * (reflected - centroid))
                f_expanded = self.func(expanded)

                if f_expanded < f_reflected:
                    simplex[worst], f_values[worst] = expanded, f_expanded
                else:
                    simplex[worst], f_values[worst] = reflected, f_reflected
            
            else:
                contracted = _apply_bounds(centroid + beta * (simplex[worst] - centroid))
                f_contracted = self.func(contracted)
                if f_contracted < f_values[worst]:
                    simplex[worst], f_values[worst] = contracted, f_contracted

                else:
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[best] + sigma * (simplex[i] - simplex[best])
                        #simplex[i] = self._apply_bounds(simplex[i])
                        f_values[:] = [self.func(p) for p in simplex]

            if np.max(np.abs(f_values - f_values[best])) < tol:
                break

        return simplex[best], f_values[best]

# Define the Rosenbrock function (banana function)
def function(x: np.array, a: float = 1, b: float = 100) -> float:
    return (a - x[0])**2 + b * (x[1] - x[0]**2)**2


# Example usage with bounds
optimizer = DownhillSimplex(func=function, x0 = np.array([5, 3]))
optimizer.set_bounds((np.array([-6, -6]), np.array([6, 6])))
best_point, best_value = optimizer.optimize()
print("Best point:", best_point)
print("Best value:", best_value)