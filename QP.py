import numpy as np
from cvxopt import matrix, solvers


class QP(object):

    def __init__(self):
        super().__init__()
        self.LagrangeMultipliers = None

    def QP(self, Q: np.array, c: np.array, A: np.array, b: np.array) -> np.array:
        """Minimize_x 1/2*x^T*Q*x+c^T*x subject to A*x<=b for given matrices Q (symmetric) and A and given vectors c and b. """
        # long live global options, which cannot be set locally
        sol = solvers.qp(matrix(np.asfarray(Q)), matrix(np.asfarray(c)),
                         matrix(np.asfarray(A)), matrix(np.asfarray(b)), options={'show_progress': False})
        self.LagrangeMultipliers = np.array(sol['z'])
        return np.array(sol['x']).flatten()


qp = QP()
print(qp.QP(np.array([[2.0]]), np.array([[-2]]),
      np.array([[1.0]]), np.array([0.5])))
print(qp.QP(np.array([[2.0]]), np.array([[-2]]),
      np.array([[1.0]]), np.array([1.0])))
print(qp.QP(np.array([[2.0]]), np.array([[-2]]),
      np.array([[1.0]]), np.array([1.5])))
