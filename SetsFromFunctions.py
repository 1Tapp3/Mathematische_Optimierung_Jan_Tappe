import numpy as np
from DifferentiableFunction import IDifferentiableFunction, DifferentiableFunction
from Set import MultidimensionalInterval
from multimethod import multimethod
from BFGS import BFGS


class BoundedSet(MultidimensionalInterval):
    """This class models bounded set in an affine space. The bounded set is given as intersection of a MultidimensionalInterval and an inequality constraint.
    An inequality constrint is given by a differentiable functions f where x satisfies the constraint if all components of f(x) are non-positive."""

    def __init__(self, lower_bounds: np.ndarray, upper_bounds: np.ndarray, InequalityConstraints: IDifferentiableFunction):
        super().__init__(lower_bounds=lower_bounds, upper_bounds=upper_bounds)
        self._InequalityConstraints = InequalityConstraints

    def contains(self, point: np.ndarray) -> bool:
        return super().contains(point) and self.InequalityConstraints.evaluate(point).max() <= 0

    def point(self) -> np.ndarray:
        starting_point = np.array([np.random.uniform(
            self.lower_bounds[i], self.upper_bounds[i]) for i in range(self._ambient_dimension)])
        bfgs = BFGS()
        penalty_function = DifferentiableFunction.FromComposition(
            DifferentiableFunction.ReLU(dimension=self._ambient_dimension), self.InequalityConstraints)
        penalty_function = DifferentiableFunction.FromComposition(
            DifferentiableFunction.TwoNormSquared(dimension=self.InequalityConstraints.evaluate(starting_point).shape[0]), penalty_function)
        # intersect penalty function with a smaller domain by adding a suitable zero function with constraints
        penalty_function = penalty_function + DifferentiableFunction(
            name="0", domain=MultidimensionalInterval(lower_bounds=self.lower_bounds, upper_bounds=self.upper_bounds), evaluate=lambda x: np.array([0]), jacobian=lambda x: (0*x).reshape(1, -1))
        x = bfgs.Minimize(penalty_function,
                          startingpoint=starting_point, tol_x=1e-9, tol_y=1e-9, )
        if penalty_function.evaluate(x) <= 0:
            return x
        else:
            return None

    @ property
    def InequalityConstraints(self) -> IDifferentiableFunction:
        """Returns the function describing the inequality constraints of this set"""
        return self._InequalityConstraints

    @ property
    def upper_bounds(self) -> np.ndarray:
        """Returns the functions that sets the inequality constraints"""
        return self._upper_bounds

    @ multimethod
    def intersect(self: 'BoundedSet', other: 'BoundedSet') -> 'BoundedSet':
        """Intersects two BoundedSets"""
        return BoundedSet(
            lower_bounds=np.maximum(self.lower_bounds, other.lower_bounds),
            upper_bounds=np.minimum(self.upper_bounds, other.upper_bounds),
            InequalityConstraints=self.InequalityConstraints.Pairing(
                other.InequalityConstraints)
        )

    @ multimethod
    def intersect(self: 'BoundedSet', other: 'MultidimensionalInterval') -> 'BoundedSet':
        """Intersects a BoundedSet with a MultidimensionalInterval"""
        lowerBounds = self.lower_bounds
        if isinstance(other, MultidimensionalInterval):
            lowerBounds = np.maximum(lowerBounds, other.lower_bounds)
        upperBounds = self.upper_bounds
        if isinstance(other, MultidimensionalInterval):
            upperBounds = np.minimum(upperBounds, other.upper_bounds)
        return BoundedSet(
            lower_bounds=lowerBounds,
            upper_bounds=upperBounds,
            InequalityConstraints=self.InequalityConstraints
        )

    # ToDo: CartesianProducts
