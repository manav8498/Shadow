# Getting help with Shadow

Shadow is maintained by volunteers. We aim to respond to new issues and
discussions **within 7 days on a best-effort basis**. There is no paid
support tier.

## Where to ask

1. **[GitHub Discussions](https://github.com/manav8498/Shadow/discussions)** -
   questions, ideas, show-and-tell, integration help. This is the right
   place for "how do I...?"
2. **[GitHub Issues](https://github.com/manav8498/Shadow/issues)** -
   bugs and concrete feature requests. Please use the issue templates.
3. **Security vulnerabilities**, do **not** post publicly. See
   [SECURITY.md](./SECURITY.md) for private disclosure instructions via
   GitHub Security Advisories.

## Before opening an issue

- Search existing [issues](https://github.com/manav8498/Shadow/issues?q=) and
  [discussions](https://github.com/manav8498/Shadow/discussions?q=).
- Check the [CHANGELOG](./CHANGELOG.md) to see if it's already fixed.
- Reproduce on the latest release if possible.
- For bugs, fill in the full template, minimal reproduction, version
  (`shadow --version`), OS, Python / Rust version.

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md). Good first issues are labelled
[`good first issue`](https://github.com/manav8498/Shadow/labels/good%20first%20issue).

## Commercial support

Shadow is pure OSS, there is currently no commercial support entity
behind it. If that changes we'll update this page.

## Yanked / broken releases

If a release on crates.io or PyPI is broken, it is **yanked** (not deleted)
so existing lockfiles keep resolving. We publish a patch release with the
fix and the yank reason in the [CHANGELOG](./CHANGELOG.md).
