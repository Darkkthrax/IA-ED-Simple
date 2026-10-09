"""FOR TESTING THE PIPELINE ONLY, NOT THE TEAM'S ANSWERS.

Used only when exam.report.run(use_demo=True). Some constants are named K,
others C, to exercise parse_answer renaming.
"""

from __future__ import annotations

# Positive branch for block B. Mix of K and C as the free constant.
ANSWERS_DEMO: dict[str, str] = {
    "A1": "2*x**3 + K",
    "A2": "x**5 + K",
    "A3": "2*x**4 + C",
    "A4": "x**3 + x**2 + C",
    "A5": "2*x**5 - 2*x**2 + K",
    "B1": "sqrt(2*x**2 + K)",
    "B2": "sqrt(2*x**3/3 + C)",
    "B3": "sqrt(x**2/3 + K)",
    "B4": "sqrt(2*x**4 + C)",
    "B5": "(3*x**2/2 + K)**(1/3)",
    "C1": "K*exp(x**2)",
    "C2": "C*exp(x**3/3)",
    "C3": "K*exp(x**3)",
    "C4": "C*exp(x**4)",
    "C5": "K*exp(3*x**2)",
}
