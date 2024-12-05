from abc import ABC, abstractmethod
from Set import ISet, AffineSpace
import numpy as np
from typing import Callable, Union
from multimethod import multimethod


class IDifferentiableFunction(object):
    """This interface models differentiable Functions from ISet to R^n."""

    def __init__(self, name: str, domain: ISet):
        super().__init__()
        self._name = name
        self._domain = domain

    @abstractmethod
    def evaluate(self, point: np.ndarray) -> np.ndarray:
        """Evaluates the function at point.
        The parameter "point" is a vector in Double^(Domain.ambient_dimension) such that Domain.is_contained(point)=true."""
        pass

    @property
    def domain(self) -> ISet:
        """The domain of the function, i.e., the set of points at which the function can be evaluated."""
        return self._domain

    @property
    def name(self) -> str:
        """The name of the function, might be used for debugging"""
        return self._name

    @abstractmethod
    def jacobian(self, point: np.ndarray) -> np.ndarray:
        """This is the evaluated Jacobian of this function f at a point.
        The parameter "point" is a vector in Double^(Domain.ambient_dimension) such that Domain.is_contained(point)=true."""
        pass

    @multimethod
    def __add__(self, other: Union[int, float]) -> 'IDifferentiableFunction':
        """Adds the two functions value wise, where the second function is a constant"""
        return DifferentiableFunction(
            name="(" + self.name + ") + " + str(other),
            domain=self.domain,
            evaluate=lambda v: self.evaluate(v) + other,
            jacobian=lambda v: self.jacobian(v)
        )

    @multimethod
    def __add__(self, other: 'IDifferentiableFunction') -> 'IDifferentiableFunction':
        """Adds the two functions value wise"""
        return DifferentiableFunction(
            name="(" + self.name + ") + (" + other.name + ")",
            domain=self.domain.intersect(other.domain),
            evaluate=lambda v: self.evaluate(v) + other.evaluate(v),
            jacobian=lambda v: self.jacobian(v) + other.jacobian(v)
        )

    def __mul__(self, other: Union[int, float]) -> 'IDifferentiableFunction':
        """Multiplies the function by a scalar"""
        return DifferentiableFunction(
            name=str(other) + " * (" + self.name + ")",
            domain=self.domain,
            evaluate=lambda v: other * self.evaluate(v),
            jacobian=lambda v: other * self.jacobian(v)
        )

    def __pow__(self, power: int) -> 'IDifferentiableFunction':
        """Take integer exponents of a function"""
        return DifferentiableFunction(
            name="(" + self.name + ")^" + str(power),
            domain=self.domain,
            evaluate=lambda v: self.evaluate(v)**power,
            jacobian=lambda v: np.matmul(
                np.array([[power]])*self.evaluate(v)**(power-1), self.jacobian(v))
        )

    def __rmul__(self, other: Union[int, float]):
        """Multiplies the function by a scalar"""
        return self.__mul__(other)

    def __sub__(self, other: 'IDifferentiableFunction') -> 'IDifferentiableFunction':
        """Subtracts two functions value wise"""
        return self + (-1) * other

    def Pairing(self, other: 'IDifferentiableFunction') -> 'IDifferentiableFunction':
        """Returns the pairing of two functions"""
        return DifferentiableFunction(
            name="Pair(" + self.name + "," + other.name + ")",
            domain=self.domain.intersect(other.domain),
            evaluate=lambda v: np.concatenate(
                (self.evaluate(v), other.evaluate(v))),
            jacobian=lambda v: np.concatenate(
                (self.jacobian(v), other.jacobian(v)))
        )

    def CartesianProduct(self, other: 'IDifferentiableFunction') -> 'IDifferentiableFunction':
        """Returns the cartesian product of two functions"""

        proj_domain = AffineSpace(
            self.domain._ambient_dimension+other.domain._ambient_dimension)
        proj1 = DifferentiableFunction.Projection(
            domain=proj_domain, l=range(0, self.domain._ambient_dimension))
        proj2 = DifferentiableFunction.Projection(
            domain=proj_domain, l=range(self.domain._ambient_dimension, self.domain._ambient_dimension+other.domain._ambient_dimension))
        f1 = DifferentiableFunction.FromComposition(
            self, proj1)
        f2 = DifferentiableFunction.FromComposition(
            other, proj2)
        return DifferentiableFunction(
            name="CartesianProduct(" + self.name + "," + other.name + ")",
            domain=self.domain.cartesian_product(other.domain),
            evaluate=lambda v: np.concatenate(
                (f1.evaluate(v), f2.evaluate(v))),
            jacobian=lambda v: np.concatenate(
                (f1.jacobian(v), f2.jacobian(v)), axis=0)
        )


class DifferentiableFunction(IDifferentiableFunction):
    """This class models differentiable Functions from ISet to R^n, where the function and the Jacobian are given by lambdas."""

    def __init__(self, name: str, domain: ISet, evaluate: Callable[[np.ndarray], np.ndarray], jacobian: Callable[[np.ndarray], np.ndarray]):
        """ Construct a function from lambdas, for the function itself and for its derivatives
        Sadly, the type system is not strict enough to check sizes of tensors"""
        super().__init__(name=name, domain=domain)
        self._evaluate = evaluate
        self._jacobian = jacobian

    def evaluate(self, point: np.ndarray) -> np.ndarray:
        return self._evaluate(point)

    def jacobian(self, point: np.ndarray) -> np.ndarray:
        return self._jacobian(point)

    @ classmethod
    def FromComposition(cls, f: IDifferentiableFunction, g: IDifferentiableFunction) -> IDifferentiableFunction:
        """Constructs f ° g"""
        return cls(
            name="(" + f.name + ") ° (" + g.name + ")",
            domain=g.domain,
            evaluate=lambda v: f.evaluate(g.evaluate(v)),
            jacobian=lambda v: np.matmul(
                f.jacobian(g.evaluate(v)), g.jacobian(v))
        )

    @ classmethod
    def LinearMapFromMatrix(cls, A: np.array) -> IDifferentiableFunction:
        """Constructs x -> A*x"""
        return cls(
            name="linear",
            domain=AffineSpace(A.shape[1]),
            evaluate=lambda x: np.matmul(A, x),
            jacobian=lambda x: A
        )

    @classmethod
    def TranslationByVector(cls, v: np.array) -> IDifferentiableFunction:
        """Constructs x -> x+v"""
        n = v.shape[0]
        return cls(name="translation", domain=AffineSpace(n), evaluate=lambda x: x+v, jacobian=lambda x: np.eye(n))

    def __create_matrix_with_ones(rows: int, columns: int, ones_positions: list[tuple[int, int]]):
        matrix = np.zeros((rows, columns))
        row_indices, column_indices = zip(*ones_positions)
        matrix[row_indices, column_indices] = 1
        return matrix

    @ classmethod
    def Projection(cls, domain: ISet, l: list[int]) -> IDifferentiableFunction:
        return cls(
            name="projection("+str(l)+")",
            domain=domain,
            evaluate=lambda x: x[l],
            jacobian=lambda x: cls.__create_matrix_with_ones(rows=len(l), columns=domain._ambient_dimension, ones_positions=list(zip(range(len(l)), l))))

    @classmethod
    def Identity(cls, domain: ISet) -> IDifferentiableFunction:
        n = domain._ambient_dimension
        return cls(
            name="Id(n)",
            domain=domain,
            evaluate=lambda x: x,
            jacobian=lambda x: np.eye(n))

    @classmethod
    def ReLU(cls, dimension: int) -> IDifferentiableFunction:
        """Returns a ReLU function"""
        return cls(name="ReLU", domain=AffineSpace(dimension), evaluate=lambda x: np.maximum(0, x), jacobian=lambda x: np.diag(x >= 0))

    @classmethod
    def TwoNormSquared(cls, dimension: int) -> IDifferentiableFunction:
        """Returns a function computing the squared two norm of a vector"""
        return cls(name="TwoNormSquared", domain=AffineSpace(dimension), evaluate=lambda x: np.linalg.norm(x)**2, jacobian=lambda x: (2*x).reshape(1, dimension))

    @classmethod
    def Debug(cls, f: IDifferentiableFunction) -> IDifferentiableFunction:
        """Returns a modified function that prints its inputs and outputs"""
        def modified_evaluate(x):
            result = f.evaluate(x)
            print(f"{f.name} - Input: {x}, Output: {result}")
            return result
        return cls(name=f.name, domain=f.domain, evaluate=modified_evaluate, jacobian=f.jacobian)
