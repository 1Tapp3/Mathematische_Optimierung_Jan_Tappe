import unittest
import numpy as np
from Set import AffineSpace
from DifferentiableFunction import DifferentiableFunction
from BFGS import BFGS


class tests_BFGS(unittest.TestCase):

    def test_BFGS1(self):
        R = AffineSpace(1)
        f = DifferentiableFunction(
            name="x->x^2", domain=R, evaluate=lambda x: np.array([x[0]**2]), jacobian=lambda x: np.array([[2*x[0]]]))
        bfgs = BFGS()
        x = bfgs.Minimize(f, startingpoint=np.array([10.0]))
        self.assertAlmostEqual(x.item(), 0, 2)
        self.assertAlmostEqual(f.evaluate(x).item(), 0, 5)

    def test_BFGS2(self):
        R = AffineSpace(3)
        f = DifferentiableFunction(
            name="(x,y,z)->x^2+y^2+z^2", domain=R, evaluate=lambda x: np.array([x[0]**2+x[1]**2+x[2]**2]), jacobian=lambda x: np.array([[2*x[0], 2*x[1], 2*x[2]]]))
        bfgs = BFGS()
        x = bfgs.Minimize(f, startingpoint=np.array([10.0, 1, 0.1]))
        self.assertAlmostEqual(np.linalg.norm(x), 0, 2)
        self.assertAlmostEqual(f.evaluate(x).item(), 0, 5)

    def test_BFGS3(self):
        R = AffineSpace(1)
        f = DifferentiableFunction(
            name="x->x^4", domain=R, evaluate=lambda x: np.array([x[0]**4]), jacobian=lambda x: np.array([[4*x[0]**3]]))
        bfgs = BFGS()
        x = bfgs.Minimize(f, startingpoint=np.array([10.0]))
        self.assertAlmostEqual(x.item(), 0, 1)
        self.assertAlmostEqual(f.evaluate(x).item(), 0, 5)

    def test_BFGS4(self):
        R = AffineSpace(3)
        f = DifferentiableFunction(
            name="(x,y,z)->x^4+y^4+z^4", domain=R, evaluate=lambda x: np.array([x[0]**4+x[1]**4+x[2]**4]), jacobian=lambda x: np.array([[4*x[0]**3, 4*x[1]**3, 4*x[2]**3]]))
        bfgs = BFGS()
        x = bfgs.Minimize(f, startingpoint=np.array([10.0, 1, 0.1]))
        self.assertAlmostEqual(np.linalg.norm(x), 0, 1)
        self.assertAlmostEqual(f.evaluate(x).item(), 0, 5)

    def test_BFGS5(self):
        R = AffineSpace(3)
        f = DifferentiableFunction(
            name="(x,y,z)->1000*x^4+y^4+0.001*z^4", domain=R, evaluate=lambda x: np.array([1000*x[0]**4+x[1]**4+0.001*x[2]**4]), jacobian=lambda x: np.array([[4000*x[0]**3, 4*x[1]**3, 0.004*x[2]**3]]))
        bfgs = BFGS()
        x = bfgs.Minimize(f, startingpoint=np.array([10.0, 10.0, 10.0]))
        self.assertAlmostEqual(np.linalg.norm(x), 0, None, "", 1.0)
        self.assertAlmostEqual(f.evaluate(x).item(), 0, 2)

    def test_BFGS6(self):
        # Defining a function this way is ugly, as we need to do mannual computations. However, it should still work.
        R = AffineSpace(2)
        f = DifferentiableFunction(
            name="Himmelblau", domain=R, evaluate=lambda x: np.array([(x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2]), jacobian=lambda x: np.array([[4*(x[0]**2 + x[1] - 11)*x[0] + 2*x[1]**2 + 2*x[0] - 14, 2*x[0]**2 + 2*x[1] - 22 + 4*(x[1]**2 + x[0] - 7)*x[1]]]))
        bfgs = BFGS()
        x = bfgs.Minimize(f, startingpoint=np.array([10.0, 10.0]))
        self.assertAlmostEqual(f.evaluate(x).item(), 0, 5)
        x = bfgs.Minimize(f, startingpoint=np.array([1.0, 1.0]))
        self.assertAlmostEqual(f.evaluate(x).item(), 0, 5)
        x = bfgs.Minimize(f, startingpoint=np.array([-10.0, 10.0]))
        self.assertAlmostEqual(f.evaluate(x).item(), 0, 5)
        x = bfgs.Minimize(f, startingpoint=np.array([10.0, -10.0]))
        self.assertAlmostEqual(f.evaluate(x).item(), 0, 5)
        x = bfgs.Minimize(f, startingpoint=np.array([-10.0, -10.0]))
        self.assertAlmostEqual(f.evaluate(x).item(), 0, 5)
        x = bfgs.Minimize(f, startingpoint=np.array([-1.0, -1.0]))
        self.assertAlmostEqual(f.evaluate(x).item(), 0, 5)
        x = bfgs.Minimize(f, startingpoint=np.array([2.0, -2.0]))
        self.assertAlmostEqual(f.evaluate(x).item(), 0, 5)
        x = bfgs.Minimize(f, startingpoint=np.array([-2.0, 2.0]))
        self.assertAlmostEqual(f.evaluate(x).item(), 0, 5)
        x = bfgs.Minimize(f, startingpoint=np.array(
            [-0.270845, -0.923039]))  # start at local maximum
        self.assertAlmostEqual(f.evaluate(x).item(), 0, 5)

    def test_BFGS7(self):
        # This is much nicer. Derivatives are computed automatically. Manual error in derivative computations are almost impossible.
        R = AffineSpace(2)
        X = DifferentiableFunction(
            name="x", domain=R, evaluate=lambda x: np.array([x[0]]), jacobian=lambda x: np.array([[1, 0]]))
        Y = DifferentiableFunction(
            name="y", domain=R, evaluate=lambda x: np.array([x[1]]), jacobian=lambda x: np.array([[0, 1]]))
        f = (X**2+Y-11)**2+(X+Y**2-7)**2
        bfgs = BFGS()
        x = bfgs.Minimize(f, startingpoint=np.array([10.0, 10.0]))
        self.assertAlmostEqual(f.evaluate(x).item(), 0, 5)
        x = bfgs.Minimize(f, startingpoint=np.array([1.0, 1.0]))
        self.assertAlmostEqual(f.evaluate(x).item(), 0, 5)
        x = bfgs.Minimize(f, startingpoint=np.array([-10.0, 10.0]))
        self.assertAlmostEqual(f.evaluate(x).item(), 0, 5)
        x = bfgs.Minimize(f, startingpoint=np.array([10.0, -10.0]))
        self.assertAlmostEqual(f.evaluate(x).item(), 0, 5)
        x = bfgs.Minimize(f, startingpoint=np.array([-10.0, -10.0]))
        self.assertAlmostEqual(f.evaluate(x).item(), 0, 5)
        x = bfgs.Minimize(f, startingpoint=np.array([-1.0, -1.0]))
        self.assertAlmostEqual(f.evaluate(x).item(), 0, 5)
        x = bfgs.Minimize(f, startingpoint=np.array([2.0, -2.0]))
        self.assertAlmostEqual(f.evaluate(x).item(), 0, 5)
        x = bfgs.Minimize(f, startingpoint=np.array([-2.0, 2.0]))
        self.assertAlmostEqual(f.evaluate(x).item(), 0, 5)
        x = bfgs.Minimize(f, startingpoint=np.array(
            [-0.270845, -0.923039]))  # start at local maximum
        self.assertAlmostEqual(f.evaluate(x).item(), 0, 5)


if __name__ == '__main__':
    unittest.main()
