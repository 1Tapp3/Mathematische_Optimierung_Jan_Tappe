import unittest
import numpy as np
from DownhillSimplex import DownhillSimplex, rosenbrock
import time

class TestDownhillSimplex(unittest.TestCase):
    def setUp(self):
        def rosenbrock(x):
            a = 1.0
            b = 100.0
            return sum(b * (x[1:] - x[:-1]**2)**2 + (a - x[:-1])**2)
        
        self.x0 = np.array([-6, 4])
        self.optimizer = DownhillSimplex(rosenbrock, self.x0)

        self.assertTrue(np.array_equal(self.optimizer.x0, self.x0))
        self.assertTrue(np.array_equal(self.optimizer.lower_bounds, -np.inf * np.ones_like(self.x0)))
        self.assertTrue(np.array_equal(self.optimizer.upper_bounds, np.inf * np.ones_like(self.x0)))

    def test_set_bounds(self):
        lower_bounds = np.array([-5.0, -5.0])
        upper_bounds = np.array([5.0, 5.0])
        self.optimizer.set_bounds((lower_bounds, upper_bounds))
        
        self.assertTrue(np.array_equal(self.optimizer.lower_bounds, lower_bounds))
        self.assertTrue(np.array_equal(self.optimizer.upper_bounds, upper_bounds))

    def test_apply_bounds(self):
        point = np.array([5, -4])
        bounded_point = self.optimizer._apply_bounds(point)
        self.assertTrue(np.array_equal(bounded_point, point))

        self.optimizer.set_bounds((np.array([0.0, 0.0]), np.array([4.0, 4.0])))
        bounded_point = self.optimizer._apply_bounds(point)
        self.assertTrue(np.array_equal(bounded_point, np.array([4.0, 0.0])))

    def test_reflect(self):
        centroid = np.array([1.0, 1.0])
        worst = np.array([2.0, 2.0])
        alpha = 1.0
        reflected, f_reflected = self.optimizer._reflect(centroid, worst, alpha)

        self.assertTrue(np.array_equal(reflected, np.array([0.0, 0.0])))
        self.assertEqual(f_reflected, rosenbrock(reflected))

    def test_expand(self):
        centroid = np.array([1, 1])
        reflected = np.array([2, 2])
        expanded, f_expanded= self.optimizer._expand(centroid, reflected, 2.0)

        self.assertTrue(np.array_equal(expanded, np.array([3.0, 3.0])))
        self.assertEqual(f_expanded, rosenbrock(expanded))

    def test_contract(self):
        centroid = np.array([1.0, 1.0])
        worst = np.array([2.0, 2.0])
        beta = 0.5
        contracted, f_contracted = self.optimizer._contract(centroid, worst, beta)

        self.assertTrue(np.array_equal(contracted, np.array([1.5, 1.5])))
        self.assertEqual(f_contracted, rosenbrock(contracted))

    def test_shrink(self):
        simplex = np.array([[1.0, 1.0], [2.0, 2.0], [0.5, 0.5]])
        best = 0
        sigma = 0.5
        shrunk_simplex = self.optimizer._shrink(simplex, best, sigma)
        
        expected_simplex = np.array([[1.0, 1.0], [1.5, 1.5], [0.75, 0.75]])
        self.assertTrue(np.allclose(shrunk_simplex, expected_simplex, atol=1e-3))

    def test_optimize(self):
        result, value = self.optimizer.optimize()

        self.assertTrue(np.allclose(result, np.array([1.0, 1.0]), atol=1e-3))
        self.assertAlmostEqual(value, 0.0, delta=1e-3)

    def test_highDim(self):
        dim =150
        x0 = np.random.uniform(-6.0,7.0, size=dim)

        optimizer = DownhillSimplex(rosenbrock, x0)
        optimizer.set_bounds((np.full(dim, -5.0), np.full(dim, 5.0)))

        start_time = time.time()
        result, value = optimizer.optimize()
        end_time = time.time()
        
        print(f"Optimization completed in {end_time - start_time:.2f} seconds.")
        print("Best point (truncated):", result[:5], "...")
        print("Best value:", value)

        self.assertAlmostEqual(value, 0.0, delta=1e-3)

if __name__ == '__main__':
    unittest.main()
