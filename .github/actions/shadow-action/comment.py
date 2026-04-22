"""Post (or update) a shadow behavioral-diff PR comment.

Uses only the Python stdlib (urllib + argparse) so the action doesn't
need extra pip installs on top of shadow. Finds the existing bot
comment by matching a hidden HTML marker and updates it in place;
creates a new comment if no existing one is found.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="owner/repo")
    parser.add_argument("--pr", required=True, help="PR number")
    parser.add_argument("--body-file", required=True, help="Markdown file to post")
    parser.add_argument(
        "--marker",
        default="shadow-behavioral-diff",
        help="Hidden HTML marker used to find & update the existing comment",
    )
    args = parser.parse_args(argv)

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        print("GITHUB_TOKEN not set", file=sys.stderr)
        return 1

    with open(args.body_file, encoding="utf-8") as f:
        body = f.read()
    marker_comment = f"<!-- {args.marker} -->"
    if marker_comment not in body:
        body = f"{marker_comment}\n{body}"

    base = f"https://api.github.com/repos/{args.repo}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "shadow-behavioral-diff-action",
    }

    existing = _find_existing_comment(base, args.pr, marker_comment, headers)
    if existing is None:
        url = f"{base}/issues/{args.pr}/comments"
        _request("POST", url, headers, json.dumps({"body": body}).encode("utf-8"))
        print(f"posted new comment on PR #{args.pr}")
    else:
        url = f"{base}/issues/comments/{existing}"
        _request("PATCH", url, headers, json.dumps({"body": body}).encode("utf-8"))
        print(f"updated existing comment #{existing} on PR #{args.pr}")
    return 0


def _find_existing_comment(
    base: str, pr: str, marker: str, headers: dict[str, str]
) -> int | None:
    url = f"{base}/issues/{pr}/comments?per_page=100"
    try:
        data = _request("GET", url, headers)
    except urllib.error.HTTPError as e:
        print(f"warning: could not list comments: {e}", file=sys.stderr)
        return None
    comments = json.loads(data)
    for comment in comments:
        if marker in comment.get("body", ""):
            return int(comment["id"])
    return None


def _request(
    method: str, url: str, headers: dict[str, str], data: bytes | None = None
) -> bytes:
    req = urllib.request.Request(url, data=data, headers=headers, method=method)  # noqa: S310
    with urllib.request.urlopen(req) as resp:  # noqa: S310
        return bytes(resp.read())


if __name__ == "__main__":
    raise SystemExit(main())
