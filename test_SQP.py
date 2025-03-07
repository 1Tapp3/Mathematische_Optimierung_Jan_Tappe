import unittest
import numpy as np
import math
import itertools
from Set import AffineSpace
from SetsFromFunctions import BoundedSet
from DifferentiableFunction import DifferentiableFunction
from SQP import SQP


class tests_SQP(unittest.TestCase):

    def test_SQP1(self):
        sqp = SQP()

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

        f = (X-10)**2+(Y-10)**2
        x = sqp.Minimize(f, startingpoint=np.array([0.1, 1.0]))
        self.assertTrue(f.domain.contains(x))
        self.assertAlmostEqual(
            np.linalg.norm(x-np.array([math.sqrt(1.5), math.sqrt(1.5)])), 0, 0)
        x = sqp.Minimize(f, startingpoint=np.array([-1.0, 1.0]))
        self.assertAlmostEqual(
            np.linalg.norm(x-np.array([math.sqrt(1.5), math.sqrt(1.5)])), 0, 0)
        x = sqp.Minimize(f, startingpoint=np.array([0.0, 0.0]))
        self.assertAlmostEqual(
            np.linalg.norm(x-np.array([math.sqrt(1.5), math.sqrt(1.5)])), 0, 0)

    def test_SQP2(self):
        sqp = SQP()

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

        f = (X+10)**2+(Y-10)**2
        x = sqp.Minimize(f, startingpoint=np.array([0.1, 1.0]))
        self.assertTrue(f.domain.contains(x))
        self.assertAlmostEqual(
            np.linalg.norm(x-np.array([-math.sqrt(1.5), math.sqrt(1.5)])), 0, 0)
        x = sqp.Minimize(f, startingpoint=np.array([-1.0, 1.0]))
        self.assertAlmostEqual(
            np.linalg.norm(x-np.array([-math.sqrt(1.5), math.sqrt(1.5)])), 0, 0)
        x = sqp.Minimize(f, startingpoint=np.array([0.0, 0.0]))
        self.assertAlmostEqual(
            np.linalg.norm(x-np.array([-math.sqrt(1.5), math.sqrt(1.5)])), 0, 0)

    def test_SQP3(self):
        sqp = SQP()

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

        f = (X)**2+(Y-10)**2
        x = sqp.Minimize(f, startingpoint=np.array([0.1, 1.0]))
        self.assertTrue(f.domain.contains(x))
        self.assertAlmostEqual(
            np.linalg.norm(x-np.array([0.0, math.sqrt(3)])), 0, 3)
        x = sqp.Minimize(f, startingpoint=np.array([-1.0, 1.0]))
        self.assertAlmostEqual(
            np.linalg.norm(x-np.array([0.0, math.sqrt(3)])), 0, 3)
        x = sqp.Minimize(f, startingpoint=np.array([0.0, 0.0]))
        self.assertAlmostEqual(
            np.linalg.norm(x-np.array([0.0, math.sqrt(3)])), 0, 3)

    def test_Himmelblau_restricted(self):
        sqp = SQP()

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

        startingpoints = [np.array(list(a)) for a in list(itertools.product(
            [-1.0, -0.5, 0.0, 0.5, 1.0], [-1.0, -0.5, 0.0, 0.5, 1.0]))]

        for startingpoint in startingpoints:
            x = sqp.Minimize(f, startingpoint=startingpoint)
            self.assertTrue(f.domain.contains(x))
            self.assertAlmostEqual(np.linalg.norm(x), math.sqrt(3), 3)

    def test_Himmelblau_full(self):
        sqp = SQP()

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

        startingpoints = [np.array(list(a)) for a in list(itertools.product(
            [-1.0, -0.5, 0.0, 0.5, 1.0], [-1.0, -0.5, 0.0, 0.5, 1.0]))]

        for startingpoint in startingpoints:
            x = sqp.Minimize(f, startingpoint=startingpoint, rho=4, rho_scale=4)
            self.assertTrue(f.domain.contains(x))
            self.assertAlmostEqual(f.evaluate(x).item(), 0, 2)
            self.assertTrue(True in [np.linalg.norm(x-result) <
                            1e-2 for result in results])

    def test_boundary_optimum(self):
        sqp = SQP()
        
        R = AffineSpace(2)       
        domain = BoundedSet(lower_bounds=np.array([0.5, 0.0]), upper_bounds=np.array([1, 1]), 
                             InequalityConstraints=DifferentiableFunction(
                             name="boundary", domain=R, 
                             evaluate=lambda x: np.array([x[0] + x[1] - 1]), 
                             jacobian=lambda x: np.array([[1, 1]])))
        
        f = DifferentiableFunction(
            name="objective", domain=domain, 
            evaluate=lambda x: (x[0] - 0.5)**2 + (x[1] - 0.5)**2, 
            jacobian=lambda x: np.array([2*(x[0] - 0.5), 2*(x[1] - 0.5)]))
        
        x_opt = sqp.Minimize(f, startingpoint=np.array([1.0, 0.0]))
        print(x_opt)
        y = f.evaluate(x_opt)
        print(y)
        self.assertAlmostEqual(y, 0.0, 4)


if __name__ == '__main__':
    unittest.main()
