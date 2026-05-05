# Getting help with Shadow

**Question?** [GitHub Discussions](https://github.com/manav8498/Shadow/discussions). **Bug?** [GitHub Issues](https://github.com/manav8498/Shadow/issues). **Vulnerability?** [private security advisory](https://github.com/manav8498/Shadow/security/advisories/new) (see [SECURITY.md](./SECURITY.md)).

We aim to respond within **7 days** on a best-effort basis. Shadow is volunteer-maintained — there is no paid support tier today.

## Where to ask

| You have | Go here |
|---|---|
| A "how do I…?" question, integration help, or a feature idea you want to discuss before opening an issue | [GitHub Discussions](https://github.com/manav8498/Shadow/discussions) |
| A reproducible bug or a concrete feature request | [GitHub Issues](https://github.com/manav8498/Shadow/issues) — please use the templates |
| A security vulnerability | Open a [private GitHub Security Advisory](https://github.com/manav8498/Shadow/security/advisories/new). Don't post publicly; see [SECURITY.md](./SECURITY.md) |
| Something governance-related (CoC violation, maintainership, trademark) | See [GOVERNANCE.md](./GOVERNANCE.md) and [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md) |

## Before opening an issue

A 60-second checklist that prevents most "could you give us a repro?" round-trips:

- Search existing [issues](https://github.com/manav8498/Shadow/issues?q=) and [discussions](https://github.com/manav8498/Shadow/discussions?q=) for the same symptom.
- Check the [CHANGELOG](./CHANGELOG.md) — the bug may already be fixed.
- Reproduce on the latest release (`pip install --upgrade shadow-diff`) if you can.
- For bugs, fill the template: minimal repro, output of `shadow --version`, OS + Python / Rust version.

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md). Good first issues are labelled [`good first issue`](https://github.com/manav8498/Shadow/labels/good%20first%20issue) — they're scoped, the failure mode is described, and the test you need to write to land the fix is named.

## Commercial support

Shadow is pure open source; there is no commercial support entity behind it today. If that changes, this page will be updated.

## Yanked or broken releases

A broken release on PyPI / npm / crates.io is **yanked**, not deleted, so existing lockfiles keep resolving. We publish a patch release with the fix and the yank reason in the [CHANGELOG](./CHANGELOG.md).
