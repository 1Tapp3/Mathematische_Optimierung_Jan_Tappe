import unittest
import numpy as np
from Set import AffineSpace
from DifferentiableFunction import DifferentiableFunction
from GradientDescent import GradientDescent


class tests_GD(unittest.TestCase):

    def test_GD(self):
        R = AffineSpace(1)
        f = DifferentiableFunction(
            name="x->x^2", domain=R, evaluate=lambda x: np.array([x[0]**2]), jacobian=lambda x: np.array([[2*x[0]]]))
        GD = GradientDescent()
        # this is a stupid optimization, that is not supposed to be working
        x = GD.Minimize(f, startingpoint=np.array([10.0]), learningrate=1.0)
        self.assertAlmostEqual(x.item(), 10)
        self.assertAlmostEqual(f.evaluate(x).item(), 100)
        # the next two should work
        x = GD.Minimize(f, startingpoint=np.array([10.0]), learningrate=0.5)
        self.assertAlmostEqual(x.item(), 0)
        self.assertAlmostEqual(f.evaluate(x).item(), 0)
        x = GD.Minimize(f, startingpoint=np.array(
            [10.0]), learningrate=0.1, iterations=1000)
        self.assertAlmostEqual(x.item(), 0)
        self.assertAlmostEqual(f.evaluate(x).item(), 0)


if __name__ == '__main__':
    unittest.main()
