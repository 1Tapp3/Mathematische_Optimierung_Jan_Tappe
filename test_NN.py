import unittest
import itertools
import numpy as np
from FeedForwardNN import ReLUFeedForwardNN
from GradientDescent import GradientDescent


class tests_NN(unittest.TestCase):

    def test_NN1(self):
        nn = ReLUFeedForwardNN()
        data_x = np.array([[a, 0]
                          for a in list(range(-5, 5+1))]).reshape((-1, 2))
        data_y = (0.2*(np.sum(data_x, axis=1))**2-2.5).reshape(-1)
        loss = nn.ToLoss(data_x, data_y)

        gd = GradientDescent()
        params = nn.params
        params = gd.Minimize(function=loss, startingpoint=params,
                             iterations=1000, learningrate=1e-2)
        nn.params = params

        sum_loss = 0
        for i in range(data_x.shape[0]):
            sum_loss = sum_loss + \
                (nn.ToFunction().evaluate(data_x[i, :])-data_y[i])**2
        mean_loss = sum_loss / data_x.shape[0]

        self.assertAlmostEqual(loss.evaluate(
            nn.params).item(), mean_loss.item())
        self.assertTrue(loss.evaluate(nn.params) < 1)

    def test_NN2(self):
        nn = ReLUFeedForwardNN()
        data_x = np.array([[0.2*a, -0.8*a]
                           for a in list(range(-5, 5+1))]).reshape((-1, 2))
        data_y = (0.2*(np.sum(data_x, axis=1))**2-2.5).reshape(-1)
        loss = nn.ToLoss(data_x, data_y)

        gd = GradientDescent()
        params = nn.params
        params = gd.Minimize(function=loss, startingpoint=params,
                             iterations=1000, learningrate=1e-2)
        nn.params = params

        sum_loss = 0
        for i in range(data_x.shape[0]):
            sum_loss = sum_loss + \
                (nn.ToFunction().evaluate(data_x[i, :])-data_y[i])**2
        mean_loss = sum_loss / data_x.shape[0]

        self.assertAlmostEqual(loss.evaluate(
            nn.params).item(), mean_loss.item())
        self.assertTrue(loss.evaluate(nn.params) < 1)

    def test_NN3(self):
        nn = ReLUFeedForwardNN()
        data_x = np.array(list(itertools.product(
            range(-5, 5+1), range(-5, 5+1)))).reshape((-1, 2))
        data_y = np.sin(data_x[:, 0]+data_x[:, 1]).reshape(-1)
        loss = nn.ToLoss(data_x, data_y)

        gd = GradientDescent()
        params = nn.params
        params = gd.Minimize(function=loss, startingpoint=params,
                             iterations=1000, learningrate=1e-2)
        nn.params = params

        sum_loss = 0
        for i in range(data_x.shape[0]):
            sum_loss = sum_loss + \
                (nn.ToFunction().evaluate(data_x[i, :])-data_y[i])**2
        mean_loss = sum_loss / data_x.shape[0]

        self.assertAlmostEqual(loss.evaluate(
            nn.params).item(), mean_loss.item())
        self.assertTrue(loss.evaluate(nn.params) < 1)


if __name__ == '__main__':
    unittest.main()
