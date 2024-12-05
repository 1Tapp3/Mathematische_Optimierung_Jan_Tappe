from abc import ABC, abstractmethod
import numpy as np
from multimethod import multimethod

# This interface models sets.


class ISet(ABC):

    def __init__(self, ambient_dimension: int):
        self._ambient_dimension = ambient_dimension

    @abstractmethod
    def contains(self, point: np.ndarray) -> bool:
        """Checks whether a point is in the set."""
        pass

    @abstractmethod
    def point(self) -> np.ndarray:
        """Returns a point in this set."""
        pass

    @property
    def ambient_dimension(self) -> int:
        """Returns the ambient dimension of this set."""
        return self._ambient_dimension


class AffineSpace(ISet):
    """This class models an affine space R^ambient_dimension."""

    def __init__(self, ambient_dimension: int):
        super().__init__(ambient_dimension=ambient_dimension)

    def contains(self, point: np.ndarray) -> bool:
        return (point.shape == (self._ambient_dimension,))

    def point(self) -> np.ndarray:
        return np.random.randn(self._ambient_dimension)

    @multimethod
    def intersect(self: 'ISet', other: 'AffineSpace') -> 'ISet':
        """Returns the intersection of this set with another affine space."""
        assert self._ambient_dimension == other._ambient_dimension, "Assume equal dimensions when intersecting sets."
        return self

    @multimethod
    def intersect(self: 'AffineSpace', other: 'ISet') -> 'ISet':
        """Returns the intersection of this set with another affine space."""
        assert self._ambient_dimension == other._ambient_dimension, "Assume equal dimensions when intersecting sets."
        return other

    @multimethod
    def cartesian_product(self: 'AffineSpace', other: 'AffineSpace') -> 'AffineSpace':
        """Build the Cartesian product between affine spaces."""
        return AffineSpace(ambient_dimension=self._ambient_dimension+other._ambient_dimension)


class MultidimensionalInterval(ISet):
    """This class models a multidimensional interval [lower_bounds[0], upper_bounds[0]] x [lower_bounds[1], upper_bounds[1]] x ..."""

    def __init__(self, lower_bounds: np.ndarray, upper_bounds: np.ndarray):
        self._lower_bounds = lower_bounds
        self._upper_bounds = upper_bounds
        assert self.lower_bounds.shape == self.upper_bounds.shape, "Both bounds must have the same shape."
        assert len(
            self.lower_bounds.shape) == 1, "Bounds must be given by a vector."
        assert (self.upper_bounds-self.lower_bounds).min(
        ) > 0, "Upper bounds must be greater than lower bounds."
        super().__init__(
            ambient_dimension=self.lower_bounds.shape[0])

    def contains(self, point: np.ndarray) -> bool:
        return (point.shape == (self._ambient_dimension,)) and (self.lower_bounds <= point).all() and (point <= self.upper_bounds).all()

    def point(self) -> np.ndarray:
        return np.array([np.random.uniform(self.lower_bounds[i], self.upper_bounds[i]) for i in range(self._ambient_dimension)])

    @ property
    def lower_bounds(self) -> np.ndarray:
        """Returns the vector of the coordinate-wise lower bounds of this set."""
        return self._lower_bounds

    @ property
    def upper_bounds(self) -> np.ndarray:
        """Returns the vector of the coordinate-wise upper bounds of this set."""
        return self._upper_bounds

    @ multimethod
    def intersect(self: 'MultidimensionalInterval', other: 'MultidimensionalInterval') -> 'MultidimensionalInterval':
        """Returns the intersection of this multidimensional interval with another multidimensional interval."""
        assert self._ambient_dimension == other._ambient_dimension, "Assume equal dimensions when intersecting sets"
        return MultidimensionalInterval(
            lower_bounds=np.maximum(self.lower_bounds, other.lower_bounds),
            upper_bounds=np.minimum(self.upper_bounds, other.upper_bounds)
        )

    @multimethod
    def cartesian_product(self: 'MultidimensionalInterval', other: 'MultidimensionalInterval') -> 'MultidimensionalInterval':
        """Build the Cartesian product between multidimensional intervals."""
        return MultidimensionalInterval(lower_bounds=np.concatenate((self.lower_bounds, other.lower_bounds)), upper_bounds=np.concatenate((self.upper_bounds, other.upper_bounds)))

    @multimethod
    def cartesian_product(self: 'AffineSpace', other: 'MultidimensionalInterval') -> 'MultidimensionalInterval':
        """Build the Cartesian product between an affine space and a multidimensional interval"""
        return MultidimensionalInterval(
            lower_bounds=np.concatenate((np.array(
                [-float.inf] * self._ambient_dimension), other.lower_bounds)),
            upper_bounds=np.concatenate((np.array([float.inf] * self._ambient_dimension), other.upper_bounds)))

    @multimethod
    def cartesian_product(self: 'MultidimensionalInterval', other: 'AffineSpace') -> 'MultidimensionalInterval':
        """Build the Cartesian product between an affine space and a multidimensional interval"""
        return MultidimensionalInterval(
            lower_bounds=np.concatenate((self.lower_bounds, np.array(
                [-float.inf] * other._ambient_dimension))),
            upper_bounds=np.concatenate((self.upper_bounds, np.array([float.inf] * other._ambient_dimension))))
