import unittest
import numpy as np
from Set import AffineSpace, MultidimensionalInterval
from SetsFromFunctions import BoundedSet
from DifferentiableFunction import DifferentiableFunction


class tests_functions(unittest.TestCase):

    def test_parabola(self):
        R = AffineSpace(1)
        f = DifferentiableFunction(
            name="x->x^2", domain=R, evaluate=lambda x: np.array([x[0]**2]), jacobian=lambda x: np.array([[2*x[0]]]))
        ff = DifferentiableFunction.FromComposition(f, f)
        fff = DifferentiableFunction.FromComposition(f, ff)
        self.assertEqual(
            f.evaluate(np.array([1])), np.array([1]))
        self.assertEqual(
            ff.evaluate(np.array([1])), np.array([1]))
        self.assertEqual(
            fff.evaluate(np.array([1])), np.array([1]))
        self.assertEqual(
            f.jacobian(np.array([1])), np.array([2]))
        self.assertEqual(
            ff.jacobian(np.array([1])), np.array([4]))
        self.assertEqual(
            fff.jacobian(np.array([1])), np.array([8]))
        self.assertEqual(
            f.evaluate(np.array([2])), np.array([4]))
        self.assertEqual(
            ff.evaluate(np.array([2])), np.array([16]))
        self.assertEqual(
            fff.evaluate(np.array([2])), np.array([256]))
        self.assertEqual(
            f.jacobian(np.array([2])), np.array([4]))
        self.assertEqual(
            ff.jacobian(np.array([2])), np.array([32]))
        self.assertEqual(
            fff.jacobian(np.array([2])), np.array([1024]))

    def test_2d(self):
        R2 = AffineSpace(2)
        f = DifferentiableFunction(name="(x,y)->(x^2*y,x*y^2)", domain=R2, evaluate=lambda x: np.array(
            [x[0]**2*x[1], x[0]*x[1]**2]), jacobian=lambda x: np.array([[2*x[0]*x[1], x[0]**2], [x[1]**2, 2*x[0]*x[1]]]))
        ff = DifferentiableFunction.FromComposition(f, f)
        fff = DifferentiableFunction.FromComposition(f, ff)
        self.assertListEqual(
            f.evaluate(np.array([1, 1])).tolist(), [1, 1])
        self.assertListEqual(
            ff.evaluate(np.array([1, 1])).tolist(), [1, 1])
        self.assertListEqual(
            fff.evaluate(np.array([1, 1])).tolist(), [1, 1])
        self.assertListEqual(
            f.evaluate(np.array([2, 2])).tolist(), [8, 8])
        self.assertListEqual(ff.evaluate(
            np.array([2, 2])).tolist(), [512, 512])
        self.assertListEqual(f.jacobian(
            np.array([2, 2])).tolist(), [[8, 4], [4, 8]])
        self.assertListEqual(f.jacobian(
            np.array([-1, 2])).tolist(), [[-4, 1], [4, -4]])
        self.assertListEqual(ff.jacobian(
            np.array([-1, 0])).tolist(), [[0, 0], [0, 0]])
        self.assertListEqual(ff.jacobian(
            np.array([0, -1])).tolist(), [[0, 0], [0, 0]])
        self.assertListEqual(ff.jacobian(np.array([2, 2])).tolist(), [
            [1280, 1024], [1024, 1280]])
        self.assertListEqual(ff.jacobian(
            np.array([-1, 2])).tolist(), [[80, -32], [-128, 80]])

    def test_2d_to_3d(self):
        R2 = AffineSpace(2)
        f = DifferentiableFunction(name="(x,y)->(x^2*y,0,x*y^2)", domain=R2, evaluate=lambda x: np.array(
            [x[0]**2*x[1], 0, x[0]*x[1]**2]), jacobian=lambda x: np.array([[2*x[0]*x[1], x[0]**2], [0, 0], [x[1]**2, 2*x[0]*x[1]]]))
        self.assertListEqual(
            f.evaluate(np.array([1, 1])).tolist(), [1, 0, 1])
        self.assertListEqual(
            f.evaluate(np.array([2, 2])).tolist(), [8, 0, 8])
        self.assertListEqual(f.jacobian(np.array([2, 2])).tolist(), [
            [8, 4], [0, 0], [4, 8]])
        self.assertListEqual(f.jacobian(
            np.array([-1, 2])).tolist(), [[-4, 1], [0, 0], [4, -4]])

    def test_2d_to_3d_to_2d(self):
        R2 = AffineSpace(2)
        f = DifferentiableFunction(name="(x,y)->(x,0,y)", domain=R2, evaluate=lambda x: np.array([
            x[0], 0, x[1]]), jacobian=lambda x: np.array([[1, 0], [0, 0], [0, 1]]))
        g = DifferentiableFunction(name="(x,y,z)->(x,z)", domain=R2, evaluate=lambda x: np.array([
            x[0]+x[1], x[2]]), jacobian=lambda x: np.array([[1, 0, 0], [0, 0, 1]]))
        fg = DifferentiableFunction.FromComposition(f, g)
        gf = DifferentiableFunction.FromComposition(g, f)
        self.assertListEqual(fg.jacobian(np.array([2, 2, 2])).tolist(), [
            [1, 0, 0], [0, 0, 0], [0, 0, 1]])
        self.assertListEqual(gf.jacobian(
            np.array([-1, 2])).tolist(), [[1, 0], [0, 1]])

    def test_componentwise_square(self):
        R2 = AffineSpace(2)
        f = DifferentiableFunction(name="(x,y)->(x^2,y^2)", domain=R2, evaluate=lambda x: np.array(
            [x[0]**2, x[1]**2]), jacobian=lambda x: np.array([[2*x[0], 0], [0, 2*x[1]]]))
        ff = DifferentiableFunction.FromComposition(f, f)
        fff = DifferentiableFunction.FromComposition(f, ff)
        self.assertListEqual(
            f.evaluate(np.array([1, 1])).tolist(), [1, 1])
        self.assertListEqual(
            ff.evaluate(np.array([1, 1])).tolist(), [1, 1])
        self.assertListEqual(
            fff.evaluate(np.array([1, 1])).tolist(), [1, 1])
        self.assertListEqual(
            f.evaluate(np.array([2, 2])).tolist(), [4, 4])
        self.assertListEqual(
            ff.evaluate(np.array([2, 2])).tolist(), [16, 16])
        self.assertListEqual(fff.evaluate(
            np.array([2, 2])).tolist(), [256, 256])
        self.assertListEqual(f.jacobian(
            np.array([2, 2])).tolist(), [[4, 0], [0, 4]])
        self.assertListEqual(f.jacobian(
            np.array([-1, 2])).tolist(), [[-2, 0], [0, 4]])
        self.assertListEqual(ff.jacobian(
            np.array([2, 2])).tolist(), [[32, 0], [0, 32]])
        self.assertListEqual(ff.jacobian(
            np.array([-1, 2])).tolist(), [[-4, 0], [0, 32]])

    def test_transpose(self):
        R2 = AffineSpace(2)
        f = DifferentiableFunction(
            name="(x,y)->(y,x)", domain=R2, evaluate=lambda x: np.array([x[1], x[0]]), jacobian=lambda x: np.array([[0, 1], [1, 0]]))
        ff = DifferentiableFunction.FromComposition(f, f)
        fff = DifferentiableFunction.FromComposition(f, ff)
        self.assertListEqual(
            f.evaluate(np.array([1, 1])).tolist(), [1, 1])
        self.assertListEqual(
            ff.evaluate(np.array([1, 1])).tolist(), [1, 1])
        self.assertListEqual(
            fff.evaluate(np.array([1, 1])).tolist(), [1, 1])
        self.assertListEqual(
            f.evaluate(np.array([2, 2])).tolist(), [2, 2])
        self.assertListEqual(
            ff.evaluate(np.array([2, 2])).tolist(), [2, 2])
        self.assertListEqual(
            fff.evaluate(np.array([2, 2])).tolist(), [2, 2])
        self.assertListEqual(
            f.evaluate(np.array([1, 2])).tolist(), [2, 1])
        self.assertListEqual(
            ff.evaluate(np.array([1, 2])).tolist(), [1, 2])
        self.assertListEqual(
            fff.evaluate(np.array([1, 2])).tolist(), [2, 1])
        self.assertListEqual(f.jacobian(
            np.array([2, 2])).tolist(), [[0, 1], [1, 0]])
        self.assertListEqual(f.jacobian(
            np.array([-1, 2])).tolist(), [[0, 1], [1, 0]])
        self.assertListEqual(ff.jacobian(
            np.array([2, 2])).tolist(), [[1, 0], [0, 1]])
        self.assertListEqual(ff.jacobian(
            np.array([-1, 2])).tolist(), [[1, 0], [0, 1]])

    def test_transpose_componentwise_square(self):
        R2 = AffineSpace(2)
        f = DifferentiableFunction(name="(x,y)->(y^2,x^2)", domain=R2, evaluate=lambda x: np.array(
            [x[1]**2, x[0]**2]), jacobian=lambda x: np.array([[0, 2*x[1]], [2*x[0], 0]]))
        ff = DifferentiableFunction.FromComposition(f, f)
        fff = DifferentiableFunction.FromComposition(f, ff)
        self.assertListEqual(
            f.evaluate(np.array([1, 1])).tolist(), [1, 1])
        self.assertListEqual(
            ff.evaluate(np.array([1, 1])).tolist(), [1, 1])
        self.assertListEqual(
            fff.evaluate(np.array([1, 1])).tolist(), [1, 1])
        self.assertListEqual(
            f.evaluate(np.array([2, 2])).tolist(), [4, 4])
        self.assertListEqual(
            ff.evaluate(np.array([2, 2])).tolist(), [16, 16])
        self.assertListEqual(fff.evaluate(
            np.array([2, 2])).tolist(), [256, 256])
        self.assertListEqual(f.jacobian(
            np.array([2, 2])).tolist(), [[0, 4], [4, 0]])
        self.assertListEqual(f.jacobian(
            np.array([-1, 2])).tolist(), [[0, 4], [-2, 0]])
        self.assertListEqual(ff.jacobian(
            np.array([2, 2])).tolist(), [[32, 0], [0, 32]])
        self.assertListEqual(ff.jacobian(
            np.array([-1, 2])).tolist(), [[-4, 0], [0, 32]])

    def test_pairing(self):
        R2 = AffineSpace(2)
        f = DifferentiableFunction(name="(x,y)->(x,2y)", domain=R2, evaluate=lambda x: np.array([
            x[0], 2*x[1]]), jacobian=lambda x: np.array([[1, 0], [0, 2]]))
        ff = f.Pairing(f)
        self.assertListEqual(ff.evaluate(
            np.array([1, 1])).tolist(), [1, 2, 1, 2])
        self.assertListEqual(ff.evaluate(
            np.array([1, 2])).tolist(), [1, 4, 1, 4])
        self.assertListEqual(ff.jacobian(np.array([1, 1])).tolist(), [
            [1, 0], [0, 2], [1, 0], [0, 2]])
        self.assertListEqual(ff.jacobian(np.array([1, 2])).tolist(), [
            [1, 0], [0, 2], [1, 0], [0, 2]])

    def test_algebra(self):
        R2 = AffineSpace(2)
        f = DifferentiableFunction(name="(x,y)->(x,2y)", domain=R2, evaluate=lambda x: np.array([
            x[0], 2*x[1]]), jacobian=lambda x: np.array([[1, 0], [0, 2]]))
        f_plus_f = f+f
        f_minus_f = f-f
        two_f = 2*f
        f_two = f*2
        two_f_minus_f_two = two_f - f_two
        self.assertListEqual(f_plus_f.evaluate(
            np.array([1, 1])).tolist(), [2, 4])
        self.assertListEqual(f_plus_f.evaluate(
            np.array([1, 2])).tolist(), [2, 8])
        self.assertListEqual(f_minus_f.evaluate(
            np.array([1, 1])).tolist(), [0, 0])
        self.assertListEqual(f_minus_f.evaluate(
            np.array([1, 2])).tolist(), [0, 0])
        self.assertListEqual(
            two_f.evaluate(np.array([1, 1])).tolist(), [2, 4])
        self.assertListEqual(
            two_f.evaluate(np.array([1, 2])).tolist(), [2, 8])
        self.assertListEqual(
            f_two.evaluate(np.array([1, 1])).tolist(), [2, 4])
        self.assertListEqual(
            f_two.evaluate(np.array([1, 2])).tolist(), [2, 8])
        self.assertListEqual(two_f_minus_f_two.evaluate(
            np.array([1, 1])).tolist(), [0, 0])
        self.assertListEqual(two_f_minus_f_two.evaluate(
            np.array([1, 2])).tolist(), [0, 0])
        self.assertListEqual(f_plus_f.jacobian(
            np.array([1, 1])).tolist(), [[2, 0], [0, 4]])
        self.assertListEqual(f_minus_f.jacobian(
            np.array([1, 1])).tolist(), [[0, 0], [0, 0]])
        self.assertListEqual(two_f.jacobian(
            np.array([1, 1])).tolist(), [[2, 0], [0, 4]])
        self.assertListEqual(f_two.jacobian(
            np.array([1, 1])).tolist(), [[2, 0], [0, 4]])
        self.assertListEqual(two_f_minus_f_two.jacobian(
            np.array([1, 1])).tolist(), [[0, 0], [0, 0]])

    def test_algebra_himmelblau(self):

        R = AffineSpace(2)
        # Defining a function this way is ugly, as we need to do mannual computations. However, it should still work.
        f_formula = DifferentiableFunction(
            name="Himmelblau", domain=R, evaluate=lambda x: np.array([(x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2]), jacobian=lambda x: np.array([[4*(x[0]**2 + x[1] - 11)*x[0] + 2*x[1]**2 + 2*x[0] - 14, 2*x[0]**2 + 2*x[1] - 22 + 4*(x[1]**2 + x[0] - 7)*x[1]]]))
        # This is much nicer. Derivatives are computed automatically. Manual error in derivative computations are almost impossible.
        X = DifferentiableFunction(
            name="x", domain=R, evaluate=lambda x: np.array([x[0]]), jacobian=lambda x: np.array([[1, 0]]))
        Y = DifferentiableFunction(
            name="y", domain=R, evaluate=lambda x: np.array([x[1]]), jacobian=lambda x: np.array([[0, 1]]))
        f_algebra = (X**2+Y-11)**2+(X+Y**2-7)**2
        # check that both functions are equal
        f = f_formula - f_algebra
        for a in [-2, -1.0, -0.5, 0, 0.1, 1.0, 2]:
            for b in [-2, -1.0, -0.5, 0, 0.1, 1.0, 2]:
                self.assertEqual(
                    f.evaluate(np.array([a, b])), 0)
                self.assertAlmostEqual(
                    f.jacobian(np.array([a, b])).sum(), 0)


if __name__ == '__main__':
    unittest.main()
