import numpy as np
from typing import Callable, Tuple


class DownhillSimplex:
    def __init__(self, func: Callable[[np.ndarray], float], x0: np.ndarray):
        """
        Initialize the Downhill Simplex optimizer.

        Parameters:
        - func: The objective function to minimize. Should take a numpy array as input and return a float.
        - x0: Startingpoint for the Algorithm
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
        return np.minimum(self.upper_bounds, np.maximum(self.lower_bounds, point))

    def _reflect(self, centroid, worst, alpha):
        reflected = self._apply_bounds(centroid + alpha * (centroid - worst))
        return reflected, self.func(reflected)

    def _expand(self, centroid, reflected, gamma):
        expanded = self._apply_bounds(centroid + gamma * (reflected - centroid))
        return expanded, self.func(expanded)

    def _contract(self, centroid, worst, beta):
        contracted = self._apply_bounds(centroid + beta * (worst - centroid))
        return contracted, self.func(contracted)

    def _shrink(self, simplex, best, sigma):
        for i in range(1, len(simplex)):
            simplex[i] = self._apply_bounds(simplex[best] + sigma * (simplex[i] - simplex[best]))
        return simplex

    
    def optimize(self, alpha=1.0, beta=0.5, gamma=2.0, sigma=0.5, max_iter=2000, tol=1e-8):
        dim = len(self.x0)
        simplex = np.empty((dim + 1, dim), dtype=float)
        f_values = np.empty(dim + 1, dtype=float)
        simplex[0] = self._apply_bounds(self.x0)
        f_values[0] = self.func(simplex[0])

        #Filling Simplex
        for i in range(dim):
            x = np.copy(self.x0)
            x[i] += 0.5 if x[i] == 0 else 0.2 * x[i]
            simplex[i + 1] = self._apply_bounds(x)
            f_values[i + 1] = self.func(simplex[i + 1])

        #Main loop
        for iteration in range(max_iter):
            worst = np.argmax(f_values)
            best = np.argmin(f_values)
            second_worst = np.argsort(f_values)[-2]
            centroid = np.mean(simplex[np.arange(dim + 1) != worst], axis=0)

            reflected, f_reflected = self._reflect(centroid, simplex[worst], alpha)
            contracted, f_contracted = self._contract(centroid, simplex[worst], beta)

            #Decision table
            action_table = [
                (f_values[best] <= f_reflected < f_values[second_worst], "reflect"),
                (f_reflected < f_values[best], "expand"),
                (f_contracted < f_values[worst], "contract"),
                (True, "shrink"),
            ]

            action = next(action_name for condition, action_name in action_table if condition)

            actions = {
                "reflect": lambda: (reflected, f_reflected),
                "expand": lambda: self._expand(centroid, reflected, gamma),
                "contract": lambda: (contracted, f_contracted),
                "shrink": lambda: self._shrink(simplex, best, sigma),
            }

            if action == "shrink":
                simplex = actions["shrink"]()
                f_values = np.array([self.func(p) for p in simplex])
            else:
                simplex[worst], f_values[worst] = actions[action]()

            #Convergence Check
            if np.max(np.abs(f_values - f_values[best])) < tol:
                break

        return simplex[best], f_values[best]


# Define the Rosenbrock function (banana function)
def rosenbrock(x: np.ndarray, a: float = 1, b: float = 100) -> float:
    return (a - x[0])**2 + b * (x[1] - x[0]**2)**2


# Example usage with bounds
optimizer = DownhillSimplex(func=rosenbrock, x0=np.array([5, -3]))
optimizer.set_bounds((np.array([-6, -6]), np.array([6, 6])))
best_point, best_value = optimizer.optimize()
print("Best point:", best_point)
print("Best value:", best_value)
