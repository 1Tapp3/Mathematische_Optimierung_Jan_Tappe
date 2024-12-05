import unittest
import numpy as np
from Set import AffineSpace, MultidimensionalInterval
from SetsFromFunctions import BoundedSet
from DifferentiableFunction import DifferentiableFunction


class tests_set(unittest.TestCase):

    def test_affine3(self):
        affine3 = AffineSpace(ambient_dimension=3)
        self.assertEqual(
            affine3._ambient_dimension, 3)
        self.assertTrue(affine3.contains(np.array([1, 2, 3])))
        self.assertFalse(
            affine3.contains(np.array([1, 2, 3, 4])))
        self.assertFalse(
            affine3.contains(np.array([[1, 2, 3]])))

    def test_affine0(self):
        affine0 = AffineSpace(ambient_dimension=0)
        self.assertEqual(
            affine0._ambient_dimension, 0)
        self.assertTrue(affine0.contains(np.array([])))
        self.assertFalse(affine0.contains(np.array([[]])))
        self.assertFalse(affine0.contains(np.array([1, 2, 3])))
        self.assertFalse(
            affine0.contains(np.array([[1, 2, 3]])))

    def test_affine_negative(self):
        affine_negative = AffineSpace(ambient_dimension=-1)
        self.assertEqual(
            affine_negative._ambient_dimension, -1)
        self.assertFalse(affine_negative.contains(np.array([])))
        self.assertFalse(
            affine_negative.contains(np.array([[]])))
        self.assertFalse(
            affine_negative.contains(np.array([1, 2, 3])))
        self.assertFalse(
            affine_negative.contains(np.array([[1, 2, 3]])))

    def test_multidimensional_interval(self):
        with self.assertRaises(AssertionError):
            MultidimensionalInterval(lower_bounds=np.array(
                [-1, -1]), upper_bounds=np.array([0, 42, 13]))
        set = MultidimensionalInterval(lower_bounds=np.array(
            [-1, -1]), upper_bounds=np.array([0, 42]))
        self.assertTrue(set.contains(np.array([-1, -1])))
        self.assertTrue(set.contains(np.array([-1.0, -1.0])))
        self.assertTrue(set.contains(np.array([-1, 42])))
        self.assertFalse(set.contains(np.array([-1, 43])))
        self.assertFalse(set.contains(np.array([-1])))
        self.assertFalse(set.contains(np.array([[-1, 42]])))

    def test_multidimensional_interval_intersection(self):
        set1 = MultidimensionalInterval(lower_bounds=np.array(
            [-1, -2]), upper_bounds=np.array([1, 42]))
        set2 = MultidimensionalInterval(lower_bounds=np.array(
            [-2, -1]), upper_bounds=np.array([42, 1]))
        set = set1.intersect(set2)
        self.assertTrue(set.contains(np.array([-1, -1])))
        self.assertTrue(set.contains(np.array([-1.0, -1.0])))
        self.assertTrue(set.contains(np.array([1, 1])))
        self.assertTrue(set.contains(np.array([0, 0])))
        self.assertTrue(set.contains(np.array([-1, 1])))
        self.assertTrue(set.contains(np.array([1, -1])))
        self.assertFalse(set.contains(np.array([-1.1, -1])))
        self.assertFalse(set.contains(np.array([-1, -1.1])))
        self.assertFalse(set.contains(np.array([1.1, 1])))
        self.assertFalse(set.contains(np.array([1, 1.1])))

    def test_bounded_set_d(self):
        R = AffineSpace(1)
        f = DifferentiableFunction(
            name="x->x^2-1", domain=R, evaluate=lambda x: np.array([x[0]**2-1]), jacobian=lambda x: np.array([[2*x[0]]]))
        set = BoundedSet(lower_bounds=np.array(
            [-2]), upper_bounds=np.array([2]), InequalityConstraints=f)
        self.assertTrue(set.contains(np.array([-1])))
        self.assertTrue(set.contains(np.array([-1.0])))
        self.assertTrue(set.contains(np.array([1])))
        self.assertTrue(set.contains(np.array([0])))
        self.assertFalse(set.contains(np.array([-1.5])))
        self.assertFalse(set.contains(np.array([-2.0])))
        self.assertFalse(set.contains(np.array([1.5])))
        self.assertFalse(set.contains(np.array([2])))

    def test_point(self):
        R = AffineSpace(1)
        self.assertTrue(R.contains(R.point()))
        S = MultidimensionalInterval(
            lower_bounds=np.array([0]), upper_bounds=np.array([1]))
        self.assertTrue(S.contains(S.point()))
        f = DifferentiableFunction(
            name="x->x^2-1", domain=R, evaluate=lambda x: np.array([x[0]**2-1]), jacobian=lambda x: np.array([[2*x[0]]]))
        T1 = BoundedSet(lower_bounds=np.array(
            [-2]), upper_bounds=np.array([2]), InequalityConstraints=f)
        self.assertTrue(T1.contains(T1.point()))
        T2 = BoundedSet(lower_bounds=np.array(
            [0]), upper_bounds=np.array([1000]), InequalityConstraints=f)
        self.assertTrue(T2.contains(T2.point()))


if __name__ == '__main__':
    unittest.main()
