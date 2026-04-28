# Shadow — Theoretical foundations

This directory contains the mathematical references for Shadow's
statistical, formal, and causal primitives. Each page explains the
guarantee a module delivers, where the guarantee comes from in the
literature, and what the practical caveats are.

These are intended for readers who want to verify Shadow's claims
against primary sources before deploying it in regulated environments
(financial services model risk, healthcare AI safety, gov / defense).

| Page | Module | Result |
|---|---|---|
| [hotelling.md](hotelling.md) | `shadow.statistical.hotelling` | Multivariate two-sample T² with OAS shrinkage and exact F-distribution null |
| [sprt.md](sprt.md) | `shadow.statistical.sprt` | Wald and mixture SPRT for sequential drift detection |
| [conformal.md](conformal.md) | `shadow.conformal` | Distribution-free split conformal + adaptive variant |
| [ltl.md](ltl.md) | `shadow.ltl` | Finite-trace LTL model checking, O(\|π\| × \|φ\|) |
| [causal.md](causal.md) | `shadow.causal` | Pearl-style do-calculus attribution |

Each page has the same structure:

1. **What it computes** — one paragraph stating the result.
2. **Guarantee** — the formal property and its conditions.
3. **Algorithm** — the procedure Shadow runs.
4. **References** — primary sources for the result.
5. **Caveats** — known limitations and where they apply.

The goal is that a reader with a stats / formal-methods background
can read a page in under five minutes and decide whether the
guarantee fits their use case.

## Why these primitives?

The combination is uncommon. LLM-eval tooling typically ships:

- LLM-as-judge rubrics (Shadow has these too — they live on the
  Judge axis).
- Bootstrap confidence intervals on per-axis means (Shadow has
  these too — see `shadow.diff.bootstrap`).
- Plain string-matching policies.

Shadow adds three layers on top:

- **Multivariate hypothesis tests** (Hotelling T²) so a regression
  on multiple axes is detected jointly, not one axis at a time.
- **Always-valid sequential testing** (mSPRT) so monitoring a canary
  doesn't accumulate Type-I error from continuous peeking.
- **Distribution-free coverage bounds** (conformal prediction) so
  the certificate carries a finite-sample PAC guarantee, not just
  a "looks fine" assertion.
- **Finite-trace temporal logic** (LTLf) so safety policies have a
  formal semantics that can be model-checked rather than
  pattern-matched.
- **Intervention-based attribution** (do-calculus, foundation-stage)
  so regression diagnosis can be causal, not correlational.

Each addresses a specific failure mode the simpler alternatives
have. The pages below explain those failure modes and the math that
fixes them.
