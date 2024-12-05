import unittest
import itertools
import numpy as np
from GP import GP


class tests_GP(unittest.TestCase):

    def test_GP(self):

        # # Maple code to reproduce this test case
        # restart;
        # with(LinearAlgebra);
        # k:=(x1,x2,y1,y2)->exp(-1/2*((x1-y1)^2+(x2-y2)^2));
        # X:=[[1,2],[3,4],[5,6]];
        # Y:=Vector([7,8,9]);
        # K:=evalf(Matrix(nops(X),nops(X),(i,j)->k(X[i][1],X[i][2],X[j][1],X[j][2])));
        # Ks:=(xs1,xs2)->Vector(nops(X),i->k(X[i][1],X[i][2],xs1,xs2));
        # alpha:=Transpose(Y).K^(-1);
        # pred:=(xs1,xs2)->alpha.Ks(xs1,xs2);
        # evalf(pred(0,0));
        # evalf(pred(2,1));
        # evalf(pred(4,3));
        # evalf(pred(6,5));
        # dpred:=(_xs1,_xs2)->subs([xs1=_xs1,xs2=_xs2],<diff(pred(xs1,xs2),xs1),diff(pred(xs1,xs2),xs2)>);
        # evalf(dpred(0,0));
        # evalf(dpred(2,1));
        # evalf(dpred(4,3));
        # evalf(dpred(6,5));
        # var:=(xs1,xs2)->k(xs1,xs2,xs1,xs2)-Transpose(Ks(xs1,xs2)).K^(-1).Ks(xs1,xs2);
        # evalf(var(0,0));
        # evalf(var(2,1));
        # evalf(var(4,3));
        # evalf(var(6,5));
        # dvar:=(_xs1,_xs2)->subs([xs1=_xs1,xs2=_xs2],<diff(var(xs1,xs2),xs1),diff(var(xs1,xs2),xs2)>);
        # evalf(dvar(0,0));
        # evalf(dvar(2,1));
        # evalf(dvar(4,3));
        # evalf(dvar(6,5));
        # sigma:=(xs1,xs2)->sqrt(var(xs1,xs2));
        # evalf(sigma(0,0));
        # evalf(sigma(2,1));
        # evalf(sigma(4,3));
        # evalf(sigma(6,5));
        # dsigma:=(_xs1,_xs2)->subs([xs1=_xs1,xs2=_xs2],<diff(sigma(xs1,xs2),xs1),diff(sigma(xs1,xs2),xs2)>);
        # evalf(dsigma(0,0));
        # evalf(dsigma(2,1));
        # evalf(dsigma(4,3));
        # evalf(dsigma(6,5));

        data_x = np.array([[1, 2], [3, 4], [5, 6]])
        data_y = np.array([7, 8, 9])
        gp = GP(data_x=data_x, data_y=data_y)
        mu = gp.PosteriorMean()
        sigma2 = gp.PosteriorVariance()
        sigma = gp.PosteriorStandardDeviation()

        self.assertAlmostEqual(mu.evaluate(
            np.array([1, 2])).item(), 7, 3)
        self.assertAlmostEqual(mu.evaluate(
            np.array([3, 4])).item(), 8, 3)
        self.assertAlmostEqual(mu.evaluate(
            np.array([5, 6])).item(), 9, 3)
        self.assertAlmostEqual(sigma2.evaluate(
            np.array([1, 2])).item(), 0, 3)
        self.assertAlmostEqual(sigma2.evaluate(
            np.array([3, 4])).item(), 0, 3)
        self.assertAlmostEqual(sigma2.evaluate(
            np.array([5, 6])).item(), 0, 3)
        self.assertAlmostEqual(sigma.evaluate(
            np.array([1, 2])).item(), 0, 2)
        self.assertAlmostEqual(sigma.evaluate(
            np.array([3, 4])).item(), 0, 2)
        self.assertAlmostEqual(sigma.evaluate(
            np.array([5, 6])).item(), 0, 2)

        self.assertAlmostEqual(mu.evaluate(
            np.array([0, 0])).item(), 0.5630289394, 3)
        self.assertAlmostEqual(mu.evaluate(
            np.array([2, 1])).item(), 2.575156089, 3)
        self.assertAlmostEqual(mu.evaluate(
            np.array([4, 3])).item(), 2.943035531, 3)
        self.assertAlmostEqual(mu.evaluate(
            np.array([6, 5])).item(), 3.310914971, 3)
        self.assertAlmostEqual(np.linalg.norm(mu.jacobian(np.array([0, 0])) -
                                              np.array([0.5630864202, 1.126115360])), 0, 3)
        self.assertAlmostEqual(np.linalg.norm(mu.jacobian(np.array([2, 1])) -
                                              np.array([-2.471226856, 2.679085323])), 0, 3)
        self.assertAlmostEqual(np.linalg.norm(mu.jacobian(np.array([4, 3])) -
                                              np.array([-2.916083740, 2.969987321])), 0, 3)
        self.assertAlmostEqual(np.linalg.norm(mu.jacobian(np.array([6, 5])) -
                                              np.array([-3.414843874, 3.206986069])), 0, 3)
        self.assertAlmostEqual(sigma2.evaluate(
            np.array([0, 0])).item(), 0.993259802354344, 3)
        self.assertAlmostEqual(sigma2.evaluate(
            np.array([2, 1])).item(), 0.864664716742376, 3)
        self.assertAlmostEqual(sigma2.evaluate(
            np.array([4, 3])).item(), 0.864664716742376, 3)
        self.assertAlmostEqual(sigma2.evaluate(
            np.array([6, 5])).item(), 0.864664716742376, 3)
        self.assertAlmostEqual(np.linalg.norm(sigma2.jacobian(np.array([0, 0])) -
                                              np.array([-0.0134803729206716, -0.0269607682119837])), 0, 3)
        self.assertAlmostEqual(np.linalg.norm(sigma2.jacobian(np.array([2, 1])) -
                                              np.array([0.270670566515277, -0.270670566515220])), 0, 3)
        self.assertAlmostEqual(np.linalg.norm(sigma2.jacobian(np.array([4, 3])) -
                                              np.array([0.270670566515249, -0.270670566515249])), 0, 3)
        self.assertAlmostEqual(np.linalg.norm(sigma2.jacobian(np.array([6, 5])) -
                                              np.array([0.270670566515220, -0.270670566515278])), 0, 3)
        self.assertAlmostEqual(sigma.evaluate(
            np.array([0, 0])).item(), 0.996624203175070, 3)
        self.assertAlmostEqual(sigma.evaluate(
            np.array([2, 1])).item(), 0.929873495020896, 3)
        self.assertAlmostEqual(sigma.evaluate(
            np.array([4, 3])).item(), 0.929873495020896, 3)
        self.assertAlmostEqual(sigma.evaluate(
            np.array([6, 5])).item(), 0.929873495020896, 3)
        self.assertAlmostEqual(np.linalg.norm(sigma.jacobian(np.array([0, 0])) -
                                              np.array([-0.00676301703175857, -0.0135260452867246])), 0, 3)
        self.assertAlmostEqual(np.linalg.norm(sigma.jacobian(np.array([2, 1])) -
                                              np.array([0.145541607522212, -0.145541607522181])), 0, 3)
        self.assertAlmostEqual(np.linalg.norm(sigma.jacobian(np.array([4, 3])) -
                                              np.array([0.145541607522196, -0.145541607522196])), 0, 3)
        self.assertAlmostEqual(np.linalg.norm(sigma.jacobian(np.array([6, 5])) -
                                              np.array([0.145541607522181, -0.145541607522212])), 0, 3)

    def test_GP_trivial(self):

        data_x = np.empty((0, 0))
        data_y = np.empty((0,))
        gp = GP(data_x=data_x, data_y=data_y)
        mu = gp.PosteriorMean()
        sigma2 = gp.PosteriorVariance()
        sigma = gp.PosteriorStandardDeviation()

        self.assertAlmostEqual(mu.evaluate(
            np.array([])).item(), 0)
        self.assertAlmostEqual(sigma2.evaluate(
            np.array([])).item(), 1)
        self.assertAlmostEqual(sigma.evaluate(
            np.array([])).item(), 1)

    def test_GP_trivial2(self):

        data_x = np.empty((0, 3))
        data_y = np.empty((0,))
        gp = GP(data_x=data_x, data_y=data_y)
        mu = gp.PosteriorMean()
        sigma2 = gp.PosteriorVariance()
        sigma = gp.PosteriorStandardDeviation()

        self.assertAlmostEqual(mu.evaluate(
            np.array([])).item(), 0)
        self.assertAlmostEqual(sigma2.evaluate(
            np.array([])).item(), 1)
        self.assertAlmostEqual(sigma.evaluate(
            np.array([])).item(), 1)


if __name__ == '__main__':
    unittest.main()
