"""TEAM answers for the 15 exam ODEs (explicit y(x) strings).

Leave each value as None until the team fills it in by hand.
The pipeline must run with missing answers (shows PENDIENTE, skips that NN).

Examples of the expected style (do NOT copy as the team's work):
  "2*x**3 + K", "sqrt(2*x**2 + K)", "K*exp(x**2)"
"""

from __future__ import annotations

from typing import Optional

# Explicit solutions; constant may be named K, C, or anything else.
ANSWERS: dict[str, Optional[str]] = {
    "A1": "2*x**3 + C",
    "A2": "x**5 + C",
    "A3": "2*x**4 + C",
    "A4": "x**3 + x**2 + C",
    "A5": "2*x**5 - 2*x**2 + C",
    "B1": "sqrt(2*x**2 + K)",
    "B2": "sqrt((2*x**3)/3 + K)",
    "B3": "sqrt((x**2)/3 + K)",
    "B4": "sqrt(2*x**4 + K)",
    "B5": "cbrt((3*x**2)/2 + K)",
    "C1": "exp(x**2 + C)",
    "C2": "exp((x**3)/3 + C)",
    "C3": "exp(x**3 + C)",
    "C4": "exp(x**4 + C)",
    "C5": "exp(3*x**2 + C)",
}
