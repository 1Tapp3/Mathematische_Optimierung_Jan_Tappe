import numpy as np
from typing import Callable, Tuple, List

#I dont use ISet or IDifferentiableFunction since DS dont need a Function to be differentiable
class DownhillSimplex:
    def __init__(self, func: Callable[[np.ndarray], float], x0: np.ndarray,):
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
        self.constrains = []

    def set_bounds(self, bounds: Tuple[np.ndarray, np.ndarray]):
        """
        Set bounds for the variables.

        Parameters:
        - bounds: A tuple of two numpy arrays (lower_bounds, upper_bounds).
        """
        self.lower_bounds, self.upper_bounds = bounds

    def set_constrains(self, constrains: List[Callable[[np.array], bool]]):
        """
        Set constrains for the variables.

        Parameters:
        - constrains: A tuple Callable Constrains.
        """
        self.constrains = constrains

    def clear(self):
        self.constrains = []
        self.lower_bounds = np.full(self.x0.shape, -np.inf)
        self.upper_bounds = np.full(self.x0.shape, np.inf)


    def _apply_bounds(self, point: np.ndarray) -> np.ndarray:
        return np.minimum(self.upper_bounds, np.maximum(self.lower_bounds, point))
    
    def _is_feasible(self, point: np.array) -> bool:
        return all(constraint(point) for constraint in self.constrains)

    def _adjust_point(self, base_point: np.array, other_point: np.array, factor: float) -> Tuple[np.array, float]:
        point = self._apply_bounds(base_point + factor * other_point)
        step_size = factor
        while not self._is_feasible(point) and step_size > 1e-6:
            step_size /= 2
            point = self._apply_bounds(base_point + step_size * other_point)
        if not self._is_feasible(point):
            return base_point, self.func(base_point)
        return point, self.func(point)

    def _reflect(self, centroid: np.array, worst: np.array, alpha) -> np.array:
        return self._adjust_point(centroid, centroid - worst, alpha)

    def _expand(self, centroid: np.array, reflected: np.array, gamma) -> np.array:
        return self._adjust_point(centroid, reflected - centroid, gamma)

    def _contract(self, centroid: np.array, worst: np.array, beta) -> np.array:
        return self._adjust_point(centroid, worst - centroid, beta)
    
    def _shrink(self, simplex: np.array, best: np.array, sigma) -> np.array:
        for i in range(1, len(simplex)):
            simplex[i], _ = self._adjust_point(simplex[best], simplex[i] - simplex[best], sigma)
        return simplex


    def optimize(self, alpha: float = 1.0, beta: float = 0.5, gamma: float = 2.0, sigma: float = 0.5, max_iter: int = 50000, tol: float = 1e-8) -> np.array:
        dim = len(self.x0)
        simplex = np.empty((dim + 1, dim), dtype=float)
        f_values = np.empty(dim + 1, dtype=float)
        simplex[0] = self._apply_bounds(self.x0)
        f_values[0] = self.func(simplex[0])

        #Filling Simplex
        for i in range(dim):
            a = 0.2
            x = np.copy(self.x0)
            x[i] += a if x[i] == 0 else a * x[i]
            simplex[i + 1] = self._apply_bounds(x)
            f_values[i + 1] = self.func(simplex[i + 1])

        #Main loop because it has less overhead than recusion in this application, loop is already readable enough for debugging
        for iteration in range(max_iter):
            worst = np.argmax(f_values)
            best = np.argmin(f_values)
            second_worst = np.argpartition(f_values, -2)[-2]
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
