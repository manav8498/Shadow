# Finite-trace Linear Temporal Logic (LTLf)

**Module:** `shadow.ltl`
**Function:** `eval_all_positions`, `check_trace`
**Compiler:** `parse_ltl`, `rule_to_ltl`

## What it computes

Given an LTLf formula φ and a finite trace π = (s_0, s_1, …, s_{n-1}),
the model checker decides whether φ holds at each position i ∈ [0, n].

Supported operators:

| Operator | Meaning at position i |
|---|---|
| `Atom(p)` | atomic predicate p evaluated against state s_i |
| `Not(φ)`  | ¬φ |
| `And(φ, ψ)` | φ ∧ ψ |
| `Or(φ, ψ)` | φ ∨ ψ |
| `Implies(φ, ψ)` | ¬φ ∨ ψ |
| `Next(φ)` | φ holds at i+1 (or False if i = n) |
| `Globally(φ)` | ∀ j ≥ i: φ holds at j |
| `Finally(φ)` | ∃ j ≥ i: φ holds at j |
| `Until(φ, ψ)` | ∃ j ≥ i: ψ@j ∧ ∀ k ∈ [i, j): φ@k (strong) |
| `WeakUntil(φ, ψ)` | (φ U ψ)@i ∨ G(φ)@i |

## Guarantee

**Decidability.** LTLf model checking on a finite trace is decidable
in O(|π| × |φ|) where |φ| is the number of subformula nodes. Shadow's
checker achieves this complexity with a bottom-up dynamic program.

**Compositionality.** Every operator's truth at position i depends only
on its children's truth at positions ≥ i. The DP fills a 2D table
(formula nodes × positions) right-to-left in linear sweeps.

## Algorithm

For each subformula φ' (in DAG order from atoms up):

- `Atom(p)`: O(|π|), one eval per position.
- `Not / And / Or / Implies`: O(|π|), elementwise on child arrays.
- `Next(φ)`: shift child array by 1, with False at the last position.
- `Globally(φ)`: right-to-left, `out[i] = φ[i] ∧ out[i+1]`. `out[n] = True`.
- `Finally(φ)`: right-to-left, `out[i] = φ[i] ∨ out[i+1]`. `out[n] = False`.
- `Until(φ, ψ)`: right-to-left, `out[i] = ψ[i] ∨ (φ[i] ∧ out[i+1])`. `out[n] = False`.
- `WeakUntil(φ, ψ)`: same recurrence as Until but `out[n] = True`.

End-of-trace semantics follow the standard "weak" LTLf convention
(Brafman & De Giacomo 2019): G(φ) is vacuously true at trace end,
F(φ) is false, and W(φ, ψ) treats φ-holding-forever as satisfying.

## Compiler — policy rule kinds → LTL formulas

`shadow.ltl.compiler.rule_to_ltl(kind, params)` translates declarative
policy rules:

| Kind | LTL formula |
|---|---|
| `no_call(tool=A)` | G(¬tool_call:A) |
| `must_call_before(first=A, then=B)` | (¬tool_call:B) W tool_call:A |
| `must_call_once(tool=A)` | F(A) ∧ G(A → X G ¬A) |
| `required_stop_reason(allowed=[r1, r2, …])` | F(stop_reason:r1 ∨ stop_reason:r2 ∨ …) |
| `forbidden_text(text=T)` | G(¬text_contains:T) |
| `must_include_text(text=T)` | F(text_contains:T) |
| `ltl_formula(formula=…)` | parse_ltl(…) |

`must_call_before` uses **weak** until so a trace that fires neither
A nor B is vacuously safe — the natural reading. Strong-until
incorrectly flags those traces as violations.

## References

- Pnueli, A. (1977). "The temporal logic of programs." FOCS 1977.
- De Giacomo, G. & Vardi, M. Y. (2013). "Linear Temporal Logic and
  Linear Dynamic Logic on Finite Traces." IJCAI 2013.
- Brafman, R. & De Giacomo, G. (2019). "Regular Decision Processes:
  A Model for Non-Markovian Domains." IJCAI 2019.

## Caveats

- The pure-Python DP is fast enough for agent-eval scale (5-100 turn
  traces, formulas with 1-20 nodes). For long-horizon protocol
  verification (10K+ turns), wrapping a C++ LTLf2DFA library
  (spot, MONA) via PyO3 would give ~100× speedup. This is tracked
  but not currently a bottleneck for any user.
- The compiler does not currently support past-time operators
  (Y, S — "yesterday", "since"). Most safety properties are
  expressible without them.
- `must_call_once` has a corner case: if A fires at the **last**
  state, the formula is satisfied because X(G(¬A)) at position n is
  vacuously true under the weak end-of-trace convention. This is
  intentional and matches the LTLf2DFA reference behavior.
