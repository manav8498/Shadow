# Shadow Lexicon

The terms Shadow uses in CLI output, panels, PR comments, briefs, and
documentation. Use this page as a reference when reading Shadow's output
or contributing changes.

## Calls

A *call* is Shadow's ship-readiness decision for a diff. Four tiers,
each a single syllable so the call reads cleanly at a glance.

| Term      | Meaning                                                              |
| --------- | -------------------------------------------------------------------- |
| **ship**  | The candidate is safe to merge. No regression detected.              |
| **hold**  | A moderate signal warrants review before merging.                    |
| **probe** | Insufficient data for a definitive call; record more pairs.          |
| **stop**  | A severe regression was detected; do not merge.                      |

## Confidence

Confidence labels accompany numerical claims. They reflect the bootstrap
CI relative to zero and the noise floor.

| Term      | Meaning                                                              |
| --------- | -------------------------------------------------------------------- |
| **firm**  | The CI excludes zero and the effect is comfortably above noise.      |
| **fair**  | The CI excludes zero but the effect is near the noise floor.         |
| **faint** | The CI crosses zero; treat the signal as directional.                |

## Artifacts

Shadow produces and consumes content-addressed artifacts. Every artifact
has a SHA-256 trace id and an immutable identity.

| Term            | Meaning                                                          |
| --------------- | ---------------------------------------------------------------- |
| **anchor**      | The baseline trace — the known-good reference.                   |
| **candidate**   | The trace under review.                                          |
| **artifact**    | Any content-addressed Shadow output (trace, diff, policy, cert). |
| **trace id**    | A short prefix of the SHA-256 content hash, used for navigation. |
| **driver**      | The change attributed as the dominant cause of a regression.     |
| **trail**       | The causal chain walking from a regressed trace back to its cause.|
| **ledger**      | The append-only record of artifacts produced by Shadow.          |
| **bundle**      | A grouped set of artifacts (e.g. a release's traces + cert).     |

## Workflow surface

| Term            | Meaning                                                          |
| --------------- | ---------------------------------------------------------------- |
| **call**        | A ship / hold / probe / stop decision derived from a diff.       |
| **brief**       | A broadcast post summarising recent state.                       |
| **holdout**     | An artifact set excluded from CI gating.                         |
| **listen**      | A long-running file-save trigger that emits ledger entries.      |
| **rebound**     | An auto-recovery action — retry, propose variant, or escalate.   |
