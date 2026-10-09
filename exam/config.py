"""Tunable exam-pipeline settings (edit here; do not hard-code elsewhere)."""

from pathlib import Path

# Arbitrary constant value plugged into explicit solutions for the NN fit.
C_VALUE = 1

# x ranges for sampling train/test points (by exercise block).
X_RANGE_A = (0.5, 3.0)  # also used for block B
X_RANGE_B = (0.5, 3.0)
X_RANGE_C = (0.0, 1.0)

# Points for the small feed-forward network (ported from the Colab demo).
N_TRAIN = 200
N_TEST = 100
EPOCHS = 3000
HIDDEN_SIZE = 64  # Linear(1, H) -> Tanh -> Linear(H, H) -> Tanh -> Linear(H, 1)
LR = 1e-3
SEED = 0

# Transformer beam search (src.solve.solve).
BEAM_SIZE = 10

# Numeric fallback for verify_explicit when simplify does not reach 0.
VERIFY_N_POINTS = 20
VERIFY_TOL = 1e-9

# Default output folder for CSV / PNG (override from Colab with a Drive path).
OUTPUT_DIR = Path("exam/output")
