<!--
Thank you for contributing. Please fill the checklist below; it's what
reviewers look at first.
-->

## What this changes

<!-- One paragraph. Why does this exist? Link issue(s) if any. -->

## Why this approach

<!-- What did you try first? What did you rule out? 2–4 sentences. -->

## Checklist

- [ ] Follows [Conventional Commits](https://www.conventionalcommits.org/)
      in commit messages (`type(scope): subject`).
- [ ] `just ci` passes locally.
- [ ] New behaviour has at least one test that fails without this PR.
- [ ] `CHANGELOG.md` updated under the unreleased section.
- [ ] If the PR touches `SPEC.md`: additive only within `0.x.y` (see
      `CONTRIBUTING.md §Writing a SPEC.md change`).
- [ ] If the PR adds a default domain heuristic (tool-name prefix list,
      refusal phrase list, bundled rubric, etc.) — **don't**. Such
      checks belong in a `Judge`. Discuss in an issue first.
- [ ] No new unpinned dependencies (Cargo + pyproject both use `=x.y.z`).
- [ ] `mypy --strict` + `cargo clippy -D warnings` + `ruff check` all clean.

## Reviewer hints

<!--
- Any part of the diff you're uncertain about?
- Any tests you'd like reviewers to run manually?
- Breaking changes or migration notes?
-->
