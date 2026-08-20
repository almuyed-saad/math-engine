import unittest

from src.engine.sympy_engine import run_sympy


class SympyEngineTests(unittest.TestCase):
    def test_derivative(self):
        result = run_sympy("Find the derivative of x^3 + 5x^2 - 3x + 7")
        self.assertEqual(result["type"], "Derivative")
        self.assertEqual(result["result"], "3*x**2 + 10*x - 3")

    def test_integral(self):
        result = run_sympy("Integrate x^2 + 3x + 2 dx")
        self.assertEqual(result["type"], "Integral")
        self.assertIn("x**3/3", result["result"])

    def test_limit(self):
        result = run_sympy("Find limit of sin(x)/x as x -> 0")
        self.assertEqual(result["type"], "Limit")
        self.assertEqual(result["result"], "1")

    def test_equation(self):
        result = run_sympy("solve x^2 + 5x + 6 = 0")
        self.assertEqual(result["type"], "Equation")
        self.assertEqual(result["result"], "[-3, -2]")

    def test_gcd(self):
        result = run_sympy("Find gcd of 84 and 30")
        self.assertEqual(result["type"], "GCD")
        self.assertIn("gcd(84,30) = 6", result["result"])

    def test_newton_raphson(self):
        result = run_sympy("Apply Newton-Raphson to x^3 - 2x - 5 = 0, x0=2, 3 iterations")
        self.assertEqual(result["type"], "Newton-Raphson")
        self.assertIn("Final answer:", result["result"])

    def test_bisection_stops_expression_at_interval_phrase(self):
        result = run_sympy("Apply bisection of x^3 - x on [0, 2], 4 iterations")
        self.assertEqual(result["type"], "Bisection")
        self.assertIn("Root", result["result"])

    def test_bisection_rejects_invalid_bracket(self):
        result = run_sympy("Apply bisection of x^2 + 1 on [-1, 1], 4 iterations")
        self.assertEqual(result["type"], "Bisection")
        self.assertIn("same sign", result["result"])


if __name__ == "__main__":
    unittest.main()
