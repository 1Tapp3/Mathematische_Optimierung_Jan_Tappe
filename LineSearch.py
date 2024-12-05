import numpy as np
from DifferentiableFunction import IDifferentiableFunction, DifferentiableFunction
from Set import AffineSpace


class LineSearch(object):
    """This class bundles approximate line search methods"""

    def __init__(self):
        super().__init__()

    # makes a univariate function out of a function suitable for line search
    # private method, hence no input checks
    def __PrepareUnivariateFunctionForLineSearch(self, function: IDifferentiableFunction, startingpoint: np.array, direction: np.array, lower_bounds: np.array = None, upper_bounds: np.array = None) -> IDifferentiableFunction:
        if lower_bounds is None:
            lower_bounds = np.full(startingpoint.shape, -np.inf)
        if upper_bounds is None:
            upper_bounds = np.full(startingpoint.shape, np.inf)
        assert lower_bounds.shape == startingpoint.shape, "bounds must have the same shape as the starting point"
        assert upper_bounds.shape == startingpoint.shape, "bounds must have the same shape as the starting point"
        assert direction.shape == startingpoint.shape, "direction must have the same shape as the starting point"
        return DifferentiableFunction(
            name="LineSearchFunctionOf(" + function.name + ")",
            domain=AffineSpace(1),
            evaluate=lambda alpha: function.evaluate(
                np.minimum(upper_bounds, np.maximum(
                    lower_bounds, startingpoint+alpha*direction))),
            jacobian=lambda alpha: np.dot(
                direction, function.jacobian(startingpoint+alpha*direction).reshape(-1))
        )

    def BasicLineSearch(self, function: IDifferentiableFunction, startingpoint: np.array, direction: np.array, step_decrease: float = 0.5, alpha=1.0) -> np.array:
        """Line search methods that tust ensures a new function value smaller than the current one."""
        assert step_decrease > 0, "the decrease must be in (0,1)"
        assert step_decrease < 1, "the decrease must be in (0,1)"
        assert alpha > 0, "the step size alpha must be positive"
        phi = self.__PrepareUnivariateFunctionForLineSearch(
            function=function, startingpoint=startingpoint, direction=direction)
        phi0 = phi.evaluate(0)
        for step in range(20):
            if phi.evaluate(alpha) < phi0:
                break
            alpha = step_decrease * alpha
        return alpha

    def BacktrackingLineSearch(self, function: IDifferentiableFunction, startingpoint: np.array, direction: np.array, step_decrease: float = 0.5, c: float = 1e-4, alpha=1.0) -> np.array:
        """BacktrackingLineSearch, ensuring the Armijo condition. Implemented as in Nocedal&Wright, Algorithm 1. Mainly relevant for (non-quasi but pure) Newton methods"""
        assert step_decrease > 0, "the decrease must be in (0,1)"
        assert step_decrease < 1, "the decrease must be in (0,1)"
        assert c > 0, "the constant c must be in (0,1)"
        assert c < 1, "the constant c must be in (0,1)"
        assert alpha > 0, "the step size alpha must be positive"

        phi = self.__PrepareUnivariateFunctionForLineSearch(
            function=function, startingpoint=startingpoint, direction=direction)
        phi0 = phi.evaluate(0)
        dphi0 = phi.jacobian(0)
        # Armijo condition
        while phi.evaluate(alpha) > phi0+c*alpha*dphi0:
            alpha = step_decrease * alpha
        return alpha

    def LineSearchForWolfeConditions(self, function: IDifferentiableFunction, startingpoint: np.array, direction: np.array, step_decrease: float = 0.5, c1: float = 1e-4, c2: float = 0.9, alpha_max=100.0, lower_bounds: np.array = None, upper_bounds: np.array = None) -> np.array:
        """Line Search for Wolfe Conditions, ensuring the strong Wolfe conditions. Implemented as in Nocedal&Wright, Algorithm 3.5"""
        assert step_decrease > 0, "the decrease must be in (0,1)"
        assert step_decrease < 1, "the decrease must be in (0,1)"
        assert c1 > 0, "the constant c1 must be in (0,1)"
        assert c1 < 1, "the constant c1 must be in (0,1)"
        assert c2 > 0, "the constant c2 must be in (0,1)"
        assert c2 < 1, "the constant c2 must be in (0,1)"
        assert c2 > c1, "we need c2>c1"
        assert alpha_max > 0, "the maximal step size alpha must be positive"

        # Set alpha and check for feasibility
        alpha_old = np.array([0.0])
        alpha = min(np.array([10.0]), 0.5*alpha_max)
        while not function.domain.contains(startingpoint+alpha*direction):
            alpha = alpha * step_decrease

        phi = self.__PrepareUnivariateFunctionForLineSearch(
            function=function, startingpoint=startingpoint, direction=direction, lower_bounds=lower_bounds, upper_bounds=upper_bounds)

        # bad for code readability, removable after adding memoization
        phi0 = phi.evaluate(0)
        dphi0 = phi.jacobian(0)
        assert dphi0 < 0, "Line search assumes negative gradient in search direction"

        phis_old = phi.evaluate(alpha)

        i = 1
        while True:
            phis = phi.evaluate(alpha)
            # Check strong Armijo condition
            if phis > phi0+c1*alpha*dphi0 or (phis > phis_old and i > 1):
                # Here, we do no longer keep the Armijo condition. Hence, a good choice is between alpha_old and alpha
                # note the order of the alphas
                return self.__ZoomForLineSearchForWolfeConditions(phi, alpha_old, alpha, c1, c2)

            dphis = phi.jacobian(alpha)
            # Check strong Wolfe conditions
            if abs(dphis) < -c2*dphi0:
                # If satisfied, return alpha
                return alpha

            # Check whether we are no longer decreasing
            if dphis > 0:
                # A local optimum is between alpha_old and alpha
                # note the (here reversed) order of the alphas
                return self.__ZoomForLineSearchForWolfeConditions(phi, alpha, alpha_old, c1, c2)
            i = i+1

            # Increase alpha (In contrast to backtracking, we always increase here. The Zoom method might decrease again.)
            alpha_old = alpha
            factor_increase = 0.1
            alpha = (1.0-factor_increase)*alpha_old+factor_increase*alpha_max
            while not function.domain.contains(startingpoint+alpha*direction):
                factor_increase = factor_increase * step_decrease
                alpha = (1.0-factor_increase)*alpha_old + \
                    factor_increase*alpha_max

            # fallback
            if i > 100 or alpha > alpha_max:
                return min(alpha, alpha_max)

            phis_old = phis

    # The following loop invariants are also conditions for the input
    # 1) of all alphas we have seen satisfying the Armijo condition, alpha1 the the one with the smallest function value
    # 2) (alpha2-alpha1) has a different sign than phi‘(alpha1). This condition ensures that the interval contains steps satisfying the strong Wolfe conditions.
    # private method, hence no input checks
    # See also https://github.com/gjkennedy/ae6310/blob/master/Line%20Search%20Algorithms.ipynb
    def __ZoomForLineSearchForWolfeConditions(self, phi: IDifferentiableFunction, alpha1: float, alpha2: float, c1: float, c2: float) -> float:
        """Submethod zoom for line search for the strong Wolfe conditions, see Algorithm 3.6 in Nocedal&Wright"""

        phi0 = phi.evaluate(0)
        dphi0 = phi.jacobian(0)
        assert dphi0 < 0, "Line search assumes negative gradient in search direction"

        step = 0
        while True:
            step = step + 1
            assert phi.evaluate(alpha1) <= phi0+c1*alpha1 * \
                dphi0, "loop invariant 1 not satisfied in step " + str(step)
            assert phi.jacobian(
                alpha1)*(alpha2-alpha1) < 0, "loop invariant 2 not satisfied in step " + str(step)

            # Keep it simple here and only use bisection
            alpha = 0.5*(alpha1+alpha2)

            phis = phi.evaluate(alpha)
            if (phis > phi0+c1*alpha*dphi0) or (phis >= phi.evaluate(alpha1)):
                # Armijo condition violated
                alpha2 = alpha
            else:
                if phi.jacobian(alpha) <= -c2*dphi0:
                    # Wolfe conditions satisfied
                    return alpha
                # Curvature condition violated
                # Make sure that we have the intervals right
                if phi.jacobian(alpha)*(alpha2-alpha1) >= 0:
                    alpha2 = alpha1
                alpha1 = alpha

            if alpha1 == alpha2:
                return alpha1

    def LineSearchForFeasibility(self, function: IDifferentiableFunction, startingpoint: np.array, direction: np.array, step_decrease: float = 0.5, alpha=1.0) -> np.array:
        """Line Search for Wolfe Conditions, ensuring the strong Wolfe conditions. Implemented as in Nocedal&Wright, Algorithm 3.5"""
        assert step_decrease > 0, "the decrease must be in (0,1)"
        assert step_decrease < 1, "the decrease must be in (0,1)"
        assert alpha > 0, "the initial step size alpha must be positive"

        while not function.domain.contains(startingpoint+alpha*direction):
            alpha = alpha * step_decrease

        return alpha
