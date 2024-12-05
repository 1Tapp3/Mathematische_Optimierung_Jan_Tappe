import unittest
import numpy as np
import math
import itertools
from Set import AffineSpace
from SetsFromFunctions import BoundedSet
from DifferentiableFunction import DifferentiableFunction
from BayesianOptimization import BO


class tests_BO(unittest.TestCase):

    def test_BO1(self):
        bo = BO()

        R = AffineSpace(2)
        X = DifferentiableFunction(
            name="x", domain=R, evaluate=lambda x: np.array([x[0]]), jacobian=lambda x: np.array([[1, 0]]))
        Y = DifferentiableFunction(
            name="y", domain=R, evaluate=lambda x: np.array([x[1]]), jacobian=lambda x: np.array([[0, 1]]))
        const = X**2+Y**2-3
        domain = BoundedSet(lower_bounds=np.array(
            [-2, -2]), upper_bounds=np.array([2, 2]), InequalityConstraints=const)
        X = DifferentiableFunction(
            name="x", domain=domain, evaluate=lambda x: np.array([x[0]]), jacobian=lambda x: np.array([[1, 0]]))
        Y = DifferentiableFunction(
            name="y", domain=domain, evaluate=lambda x: np.array([x[1]]), jacobian=lambda x: np.array([[0, 1]]))

        f = X**2+Y**2
        x = bo.Minimize(f)
        self.assertTrue(f.domain.contains(x))
        self.assertAlmostEqual(
            np.linalg.norm(x-np.array([0, 0])), 0, 0)

        f = (X-10)**2+(Y-10)**2
        x = bo.Minimize(f)
        self.assertTrue(f.domain.contains(x))
        self.assertAlmostEqual(
            np.linalg.norm(x-np.array([math.sqrt(1.5), math.sqrt(1.5)])), 0, 0)

        f = (X+10)**2+(Y-10)**2
        x = bo.Minimize(f)
        self.assertTrue(f.domain.contains(x))
        self.assertAlmostEqual(
            np.linalg.norm(x-np.array([-math.sqrt(1.5), math.sqrt(1.5)])), 0, 0)

        f = (X)**2+(Y-10)**2
        x = bo.Minimize(f)
        self.assertTrue(f.domain.contains(x))
        self.assertAlmostEqual(
            np.linalg.norm(x-np.array([0, math.sqrt(3)])), 0, 0)

    def test_Himmelblau_restricted(self):
        bo = BO()

        R = AffineSpace(2)
        X = DifferentiableFunction(
            name="x", domain=R, evaluate=lambda x: np.array([x[0]]), jacobian=lambda x: np.array([[1, 0]]))
        Y = DifferentiableFunction(
            name="y", domain=R, evaluate=lambda x: np.array([x[1]]), jacobian=lambda x: np.array([[0, 1]]))
        const = X**2+Y**2-3
        domain = BoundedSet(lower_bounds=np.array(
            [-2, -2]), upper_bounds=np.array([2, 2]), InequalityConstraints=const)
        X = DifferentiableFunction(
            name="x", domain=domain, evaluate=lambda x: np.array([x[0]]), jacobian=lambda x: np.array([[1, 0]]))
        Y = DifferentiableFunction(
            name="y", domain=domain, evaluate=lambda x: np.array([x[1]]), jacobian=lambda x: np.array([[0, 1]]))

        f = (X**2+Y-11)**2+(X+Y**2-7)**2

        x = bo.Minimize(f)
        self.assertTrue(f.domain.contains(x))
        self.assertAlmostEqual(np.linalg.norm(x), math.sqrt(3), 3)

    def test_Himmelblau_full(self):
        bo = BO()

        R = AffineSpace(2)
        X = DifferentiableFunction(
            name="x", domain=R, evaluate=lambda x: np.array([x[0]]), jacobian=lambda x: np.array([[1, 0]]))
        Y = DifferentiableFunction(
            name="y", domain=R, evaluate=lambda x: np.array([x[1]]), jacobian=lambda x: np.array([[0, 1]]))
        const = X**2+Y**2-25.0
        domain = BoundedSet(lower_bounds=np.array(
            [-4, -4]), upper_bounds=np.array([4, 4]), InequalityConstraints=const)
        X = DifferentiableFunction(
            name="x", domain=domain, evaluate=lambda x: np.array([x[0]]), jacobian=lambda x: np.array([[1, 0]]))
        Y = DifferentiableFunction(
            name="y", domain=domain, evaluate=lambda x: np.array([x[1]]), jacobian=lambda x: np.array([[0, 1]]))

        f = (X**2+Y-11)**2+(X+Y**2-7)**2
        results = [np.array(v) for v in [
            [3.0, 2.0], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.584428, -1.848126]]]

        x = bo.Minimize(f, iterations=100)
        self.assertTrue(f.domain.contains(x))
        self.assertAlmostEqual(f.evaluate(x).item(), 0, 1)
        self.assertTrue(True in [np.linalg.norm(x-result) <
                                 1e-1 for result in results])


if __name__ == '__main__':
    unittest.main()
