"""Prefix-notation tokenizer for ODE expressions (SymPy <-> tokens)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Union

import sympy as sp

# Symbols used when decoding back to SymPy.
_X = sp.Symbol("x")
_Y = sp.Symbol("y")
_C = sp.Symbol("C")

OPERATORS = ("add", "mul", "pow", "div", "neg", "log", "exp", "eq", "diff")
SPECIAL_DEFAULT = ("PAD", "SOS", "EOS", "UNK")

# Match y' / dy/dx style derivatives in ODE strings before SymPy parsing.
# Note: do not use \b after y' — apostrophe is non-word, so \b never matches there.
_DERIV_PATTERNS = (
    (re.compile(r"\by'(?!\w)"), "Derivative(y, x)"),
    (re.compile(r"\bdy/dx\b"), "Derivative(y, x)"),
)


@dataclass
class Vocab:
    """Bidirectional token <-> id vocabulary."""

    token_to_id: dict[str, int] = field(default_factory=dict)
    id_to_token: list[str] = field(default_factory=list)

    @classmethod
    def build(
        cls,
        max_int: int = 20,
        special_tokens: Sequence[str] = SPECIAL_DEFAULT,
        extra_tokens: Iterable[str] = (),
    ) -> "Vocab":
        tokens: list[str] = []
        for t in special_tokens:
            if t not in tokens:
                tokens.append(t)
        for t in OPERATORS:
            if t not in tokens:
                tokens.append(t)
        for t in ("x", "y", "C"):
            if t not in tokens:
                tokens.append(t)
        for i in range(0, max_int + 1):
            tok = str(i)
            if tok not in tokens:
                tokens.append(tok)
        for t in extra_tokens:
            if t not in tokens:
                tokens.append(t)

        token_to_id = {t: i for i, t in enumerate(tokens)}
        return cls(token_to_id=token_to_id, id_to_token=tokens)

    def __len__(self) -> int:
        return len(self.id_to_token)

    def encode_token(self, token: str) -> int:
        return self.token_to_id.get(token, self.token_to_id["UNK"])

    def decode_token(self, idx: int) -> str:
        if 0 <= idx < len(self.id_to_token):
            return self.id_to_token[idx]
        return "UNK"

    @property
    def pad_id(self) -> int:
        return self.token_to_id["PAD"]

    @property
    def sos_id(self) -> int:
        return self.token_to_id["SOS"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["EOS"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["UNK"]


@dataclass
class Tokenizer:
    """SymPy expression <-> prefix token conversion with vocabulary."""

    max_int: int = 20
    vocab: Vocab = field(default_factory=Vocab)

    def __post_init__(self) -> None:
        if not self.vocab.token_to_id:
            self.vocab = Vocab.build(max_int=self.max_int)

    # ------------------------------------------------------------------
    # Parsing user / dataset strings
    # ------------------------------------------------------------------

    def parse_ode_string(self, text: str) -> sp.Equality:
        """Parse strings like \"y' = x*exp(y)\" into a SymPy Equality."""
        s = text.strip()
        for pat, repl in _DERIV_PATTERNS:
            s = pat.sub(repl, s)
        if "=" not in s:
            raise ValueError(f"ODE string must contain '=': {text!r}")
        left, right = s.split("=", 1)
        local = {"x": _X, "y": _Y, "C": _C, "Derivative": sp.Derivative, "exp": sp.exp, "log": sp.log}
        lhs = sp.sympify(left.strip(), locals=local)
        rhs = sp.sympify(right.strip(), locals=local)
        return sp.Eq(lhs, rhs)

    def parse_solution_string(self, text: str) -> sp.Equality:
        """Parse implicit solutions like \"log(y) = x**2/2 + C\"."""
        s = text.strip()
        if "=" not in s:
            raise ValueError(f"Solution string must contain '=': {text!r}")
        left, right = s.split("=", 1)
        local = {"x": _X, "y": _Y, "C": _C, "exp": sp.exp, "log": sp.log}
        lhs = sp.sympify(left.strip(), locals=local)
        rhs = sp.sympify(right.strip(), locals=local)
        return sp.Eq(lhs, rhs)

    def parse_any(self, text: str) -> sp.Expr:
        """Parse an ODE or solution string (or a bare expression)."""
        s = text.strip()
        for pat, repl in _DERIV_PATTERNS:
            s = pat.sub(repl, s)
        local = {
            "x": _X,
            "y": _Y,
            "C": _C,
            "Derivative": sp.Derivative,
            "exp": sp.exp,
            "log": sp.log,
        }
        if "=" in s:
            left, right = s.split("=", 1)
            return sp.Eq(
                sp.sympify(left.strip(), locals=local),
                sp.sympify(right.strip(), locals=local),
            )
        return sp.sympify(s, locals=local)

    # ------------------------------------------------------------------
    # SymPy -> prefix tokens
    # ------------------------------------------------------------------

    def sympy_to_tokens(self, expr: sp.Expr) -> list[str]:
        """Convert a SymPy expression to prefix tokens."""
        expr = sp.sympify(expr)
        return self._to_prefix(expr)

    def _to_prefix(self, expr: sp.Expr) -> list[str]:
        if isinstance(expr, sp.Equality):
            return ["eq"] + self._to_prefix(expr.lhs) + self._to_prefix(expr.rhs)

        if isinstance(expr, sp.Derivative):
            # Expect y' = Derivative(y, x)
            return ["diff"] + self._to_prefix(expr.expr) + self._to_prefix(expr.variables[0])

        if isinstance(expr, sp.Add):
            return self._nary("add", expr.args)

        if isinstance(expr, sp.Mul):
            # Separate denominator via Rational / Pow with neg exp when possible.
            return self._mul_to_prefix(expr)

        if isinstance(expr, sp.Pow):
            base, exp = expr.as_base_exp()
            if exp == -1:
                return ["div", "1"] + self._to_prefix(base)
            return ["pow"] + self._to_prefix(base) + self._to_prefix(exp)

        if isinstance(expr, sp.log):
            return ["log"] + self._to_prefix(expr.args[0])

        if isinstance(expr, sp.exp):
            return ["exp"] + self._to_prefix(expr.args[0])

        if isinstance(expr, sp.Abs):
            # Convention: ignore Abs (same as generator).
            return self._to_prefix(expr.args[0])

        if isinstance(expr, sp.Rational) and not isinstance(expr, sp.Integer):
            p, q = int(expr.p), int(expr.q)
            if p < 0:
                return ["neg", "div"] + self._int_tokens(-p) + self._int_tokens(q)
            return ["div"] + self._int_tokens(p) + self._int_tokens(q)

        if isinstance(expr, sp.Integer) or (expr.is_Integer and expr.is_number):
            n = int(expr)
            if n < 0:
                return ["neg"] + self._int_tokens(-n)
            return self._int_tokens(n)

        if expr == _X or (isinstance(expr, sp.Symbol) and expr.name == "x"):
            return ["x"]
        if expr == _Y or (isinstance(expr, sp.Symbol) and expr.name == "y"):
            return ["y"]
        if expr == _C or (isinstance(expr, sp.Symbol) and expr.name == "C"):
            return ["C"]

        if isinstance(expr, sp.Symbol):
            return ["UNK"]

        # Fallback: unary functions we might meet.
        if expr.args:
            # Unknown function -> UNK + children (best-effort).
            out = ["UNK"]
            for a in expr.args:
                out.extend(self._to_prefix(a))
            return out

        return ["UNK"]

    def _int_tokens(self, n: int) -> list[str]:
        """Encode a non-negative integer; digit-level if above max_int."""
        if n <= self.max_int:
            return [str(n)]
        # Digit-level: mul of powers of 10 would be heavy; use nested add of digits.
        # Represent large n as decimal digit sequence under a dedicated scheme:
        #   digits d_k ... d_0  => nested: add(mul(d_k, pow(10,k)), ...)
        # Simpler: chain as (((d_k)*10 + d_{k-1})*10 + ... )
        digits = [int(ch) for ch in str(n)]
        tokens: list[str] = [str(digits[0])]
        for d in digits[1:]:
            tokens = ["add", "mul"] + tokens + ["10", str(d)]
        return tokens

    def _nary(self, op: str, args: tuple[sp.Expr, ...]) -> list[str]:
        """Encode n-ary Add/Mul as a left-associated binary tree."""
        if not args:
            return ["0"] if op == "add" else ["1"]
        if len(args) == 1:
            return self._to_prefix(args[0])
        tokens = [op] + self._to_prefix(args[0]) + self._to_prefix(args[1])
        for a in args[2:]:
            tokens = [op] + tokens + self._to_prefix(a)
        return tokens

    def _mul_to_prefix(self, expr: sp.Mul) -> list[str]:
        """Encode Mul, preferring div for inverse factors."""
        numer, denom = sp.fraction(sp.together(expr))
        if denom != 1:
            # Avoid rewriting when numer/denom already simple enough.
            if numer != expr or denom != 1:
                return ["div"] + self._to_prefix(numer) + self._to_prefix(denom)

        args = expr.args
        # Pull out -1 as neg.
        if sp.Integer(-1) in args:
            rest = [a for a in args if a != -1]
            if not rest:
                return ["neg", "1"]
            if len(rest) == 1:
                return ["neg"] + self._to_prefix(rest[0])
            return ["neg"] + self._nary("mul", tuple(rest))
        return self._nary("mul", args)

    # ------------------------------------------------------------------
    # Prefix tokens -> SymPy
    # ------------------------------------------------------------------

    def tokens_to_sympy(self, tokens: Sequence[str]) -> sp.Expr:
        """Parse prefix tokens into a SymPy expression."""
        expr, end = self._parse_prefix(list(tokens), 0)
        if end != len(tokens):
            raise ValueError(
                f"Trailing tokens starting at {end}: {tokens[end:]!r}"
            )
        return expr

    def _parse_prefix(self, tokens: list[str], i: int) -> tuple[sp.Expr, int]:
        if i >= len(tokens):
            raise ValueError("Unexpected end of tokens")
        tok = tokens[i]

        if tok == "eq":
            lhs, j = self._parse_prefix(tokens, i + 1)
            rhs, k = self._parse_prefix(tokens, j)
            return sp.Eq(lhs, rhs), k

        if tok == "diff":
            expr, j = self._parse_prefix(tokens, i + 1)
            var, k = self._parse_prefix(tokens, j)
            return sp.Derivative(expr, var), k

        if tok == "add":
            a, j = self._parse_prefix(tokens, i + 1)
            b, k = self._parse_prefix(tokens, j)
            return sp.Add(a, b, evaluate=False), k

        if tok == "mul":
            a, j = self._parse_prefix(tokens, i + 1)
            b, k = self._parse_prefix(tokens, j)
            return sp.Mul(a, b, evaluate=False), k

        if tok == "pow":
            a, j = self._parse_prefix(tokens, i + 1)
            b, k = self._parse_prefix(tokens, j)
            return sp.Pow(a, b, evaluate=False), k

        if tok == "div":
            a, j = self._parse_prefix(tokens, i + 1)
            b, k = self._parse_prefix(tokens, j)
            return sp.Mul(a, sp.Pow(b, -1, evaluate=False), evaluate=False), k

        if tok == "neg":
            a, j = self._parse_prefix(tokens, i + 1)
            return sp.Mul(sp.Integer(-1), a, evaluate=False), j

        if tok == "log":
            a, j = self._parse_prefix(tokens, i + 1)
            return sp.log(a), j

        if tok == "exp":
            a, j = self._parse_prefix(tokens, i + 1)
            return sp.exp(a), j

        if tok == "x":
            return _X, i + 1
        if tok == "y":
            return _Y, i + 1
        if tok == "C":
            return _C, i + 1

        if tok.isdigit() or (tok.startswith("-") and tok[1:].isdigit()):
            return sp.Integer(int(tok)), i + 1

        if tok in ("PAD", "SOS", "EOS", "UNK"):
            raise ValueError(f"Cannot parse special token as expression: {tok}")

        raise ValueError(f"Unknown token: {tok!r} at position {i}")

    # ------------------------------------------------------------------
    # High-level encode / decode
    # ------------------------------------------------------------------

    def tokenize(self, text_or_expr: Union[str, sp.Expr]) -> list[str]:
        """Tokenize an ODE/solution string or SymPy expression to prefix tokens."""
        if isinstance(text_or_expr, str):
            expr = self.parse_any(text_or_expr)
        else:
            expr = text_or_expr
        return self.sympy_to_tokens(expr)

    def encode(
        self,
        text_or_expr: Union[str, sp.Expr],
        add_special: bool = True,
    ) -> list[int]:
        """Tokenize and map to vocabulary ids (optional SOS/EOS)."""
        tokens = self.tokenize(text_or_expr)
        ids = [self.vocab.encode_token(t) for t in tokens]
        if add_special:
            return [self.vocab.sos_id] + ids + [self.vocab.eos_id]
        return ids

    def decode_ids(
        self,
        ids: Sequence[int],
        strip_special: bool = True,
    ) -> list[str]:
        tokens = [self.vocab.decode_token(i) for i in ids]
        if strip_special:
            tokens = [t for t in tokens if t not in ("PAD", "SOS", "EOS")]
            if "EOS" in [self.vocab.decode_token(i) for i in ids]:
                # Truncate at first EOS in original sequence.
                out: list[str] = []
                for i in ids:
                    tok = self.vocab.decode_token(i)
                    if tok == "EOS":
                        break
                    if tok in ("PAD", "SOS"):
                        continue
                    out.append(tok)
                return out
        return tokens

    def ids_to_sympy(self, ids: Sequence[int]) -> sp.Expr:
        tokens = self.decode_ids(ids, strip_special=True)
        return self.tokens_to_sympy(tokens)

    def round_trip(self, text_or_expr: Union[str, sp.Expr]) -> sp.Expr:
        """sympy -> tokens -> sympy (for testing)."""
        tokens = self.tokenize(text_or_expr)
        return self.tokens_to_sympy(tokens)

    def round_trip_ok(
        self,
        text_or_expr: Union[str, sp.Expr],
        *,
        numerical: bool = True,
    ) -> bool:
        """Check algebraic (and optional numeric) equivalence after round-trip."""
        if isinstance(text_or_expr, str):
            original = self.parse_any(text_or_expr)
        else:
            original = sp.sympify(text_or_expr)
        rebuilt = self.round_trip(original)
        return expressions_equivalent(original, rebuilt, numerical=numerical)


def expressions_equivalent(
    a: sp.Expr,
    b: sp.Expr,
    *,
    numerical: bool = True,
    n_points: int = 3,
) -> bool:
    """True if two expressions are symbolically (or numerically) equivalent."""
    if isinstance(a, sp.Equality) or isinstance(b, sp.Equality):
        if not (isinstance(a, sp.Equality) and isinstance(b, sp.Equality)):
            return False
        # Do not simplify the Equality itself — SymPy may drop Derivative.
        if sp.simplify(a.lhs - b.lhs) == 0 and sp.simplify(a.rhs - b.rhs) == 0:
            return True
        da = sp.simplify(a.lhs - a.rhs)
        db = sp.simplify(b.lhs - b.rhs)
        if sp.simplify(da - db) == 0 or sp.simplify(da + db) == 0:
            return True
        if numerical:
            return _numeric_eq_check(a, b, n_points=n_points)
        return False

    diff = sp.simplify(sp.expand(a) - sp.expand(b))
    if diff == 0:
        return True
    if numerical:
        return _numeric_expr_check(a - b, n_points=n_points)
    return False


def _sample_points(symbols: set[sp.Symbol], n_points: int) -> list[dict[sp.Symbol, float]]:
    # Positive reals to keep log/exp domains happier.
    import random

    rng = random.Random(0)
    pts: list[dict[sp.Symbol, float]] = []
    for _ in range(n_points):
        pts.append({s: rng.uniform(0.5, 2.5) for s in symbols})
    return pts


def _numeric_expr_check(expr: sp.Expr, n_points: int = 3, tol: float = 1e-6) -> bool:
    symbols = {s for s in expr.free_symbols if isinstance(s, sp.Symbol)}
    if not symbols:
        try:
            return abs(float(expr.evalf())) < tol
        except Exception:
            return False
    for sub in _sample_points(symbols, n_points):
        try:
            val = complex(expr.evalf(subs=sub))
            if abs(val) > tol:
                return False
        except Exception:
            return False
    return True


def _numeric_eq_check(a: sp.Equality, b: sp.Equality, n_points: int = 3, tol: float = 1e-5) -> bool:
    """Check that residual surfaces match at random points (weak fallback)."""
    ra = sp.simplify(a.lhs - a.rhs)
    rb = sp.simplify(b.lhs - b.rhs)
    symbols = {
        s
        for s in (ra.free_symbols | rb.free_symbols)
        if isinstance(s, sp.Symbol) and s.name in ("x", "y", "C")
    }
    if not symbols:
        return _numeric_expr_check(ra - rb, n_points=n_points, tol=tol)
    for sub in _sample_points(symbols, n_points):
        try:
            va = complex(ra.evalf(subs=sub))
            vb = complex(rb.evalf(subs=sub))
            # Residuals proportional (same implicit curve family) is too loose;
            # require close residuals or opposite sign-scaled match.
            if abs(va - vb) > tol and abs(va + vb) > tol:
                return False
        except Exception:
            return False
    return True


def build_tokenizer_from_config(cfg: Optional[dict] = None) -> Tokenizer:
    """Create a Tokenizer from a config mapping (tokenizer: block or flat)."""
    cfg = cfg or {}
    block = cfg.get("tokenizer", cfg)
    max_int = int(block.get("max_int", 20))
    special = tuple(block.get("special_tokens", SPECIAL_DEFAULT))
    vocab = Vocab.build(max_int=max_int, special_tokens=special)
    return Tokenizer(max_int=max_int, vocab=vocab)
