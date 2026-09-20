#!/usr/bin/env python3
"""Parse an issue title and select architecture + source context for triage.

Reads docs/agents/area-map.json (stdlib json only). Intended for:

- GitHub Agentic Workflow pre-activation steps (writes GITHUB_OUTPUT)
- Local checks and unit tests
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAP = REPO_ROOT / "docs" / "agents" / "area-map.json"


def load_area_map(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_title(title: str, pattern: str) -> tuple[str, str, str] | None:
    match = re.match(pattern, title.strip())
    if not match:
        return None
    return match.group(1).lower(), match.group(2).lower(), match.group(3).strip()


def resolve_area(area: str, area_map: dict[str, Any]) -> str:
    aliases = area_map.get("aliases") or {}
    resolved = aliases.get(area, area)
    if resolved not in area_map.get("areas", {}):
        return resolved
    return resolved


def select_context(title: str, area_map: dict[str, Any]) -> dict[str, Any]:
    parsed = parse_title(title, area_map["title_pattern"])
    if parsed is None:
        return {
            "valid": False,
            "issue_type": "",
            "issue_area": "",
            "kind_label": "",
            "area_label": "",
            "extra_labels": [],
            "layer": "",
            "notes": "",
            "context_files": list(area_map.get("always_read") or []),
        }

    issue_type, issue_area, _summary = parsed
    type_meta = (area_map.get("types") or {}).get(issue_type, {})
    resolved_area = resolve_area(issue_area, area_map)
    area_meta = (area_map.get("areas") or {}).get(resolved_area, {})

    always_read = list(area_map.get("always_read") or [])
    area_files = list(area_meta.get("files") or [])
    context_files: list[str] = []
    for path in always_read + area_files:
        if path not in context_files:
            context_files.append(path)

    extra_labels = list(type_meta.get("extra_labels") or [])
    area_label = area_meta.get("label") or ""
    if area_label and area_label not in extra_labels:
        extra_labels.append(area_label)

    return {
        "valid": True,
        "issue_type": issue_type,
        "issue_area": resolved_area,
        "kind_label": type_meta.get("kind_label") or "",
        "area_label": area_label,
        "extra_labels": extra_labels,
        "layer": area_meta.get("layer") or "",
        "notes": area_meta.get("notes") or "",
        "context_files": context_files,
    }


def write_github_output(result: dict[str, Any], output_path: str) -> None:
    files = "\n".join(result["context_files"])
    extra = ",".join(result["extra_labels"])
    with open(output_path, "a", encoding="utf-8") as handle:
        handle.write(f"valid={'true' if result['valid'] else 'false'}\n")
        handle.write(f"issue_type={result['issue_type']}\n")
        handle.write(f"issue_area={result['issue_area']}\n")
        handle.write(f"kind_label={result['kind_label']}\n")
        handle.write(f"area_label={result['area_label']}\n")
        handle.write(f"extra_labels={extra}\n")
        handle.write(f"layer={result['layer']}\n")
        handle.write("context_files<<EOF\n")
        handle.write(f"{files}\n")
        handle.write("EOF\n")
        handle.write("notes<<EOF\n")
        handle.write(f"{result['notes']}\n")
        handle.write("EOF\n")


def emit_stdout(result: dict[str, Any]) -> None:
    payload = dict(result)
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")


def missing_paths(result: dict[str, Any], root: Path) -> list[str]:
    missing = []
    for rel in result["context_files"]:
        if not (root / rel).exists():
            missing.append(rel)
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--title", default=os.environ.get("ISSUE_TITLE", ""))
    parser.add_argument("--map", dest="map_path", default=str(DEFAULT_MAP))
    parser.add_argument("--check-paths", action="store_true")
    args = parser.parse_args()

    if not args.title:
        print("ISSUE_TITLE / --title is required", file=sys.stderr)
        return 2

    area_map = load_area_map(Path(args.map_path))
    result = select_context(args.title, area_map)

    if args.check_paths:
        missing = missing_paths(result, REPO_ROOT)
        if missing:
            print("missing context files:", file=sys.stderr)
            for path in missing:
                print(f"  {path}", file=sys.stderr)
            return 1

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        write_github_output(result, github_output)
    else:
        emit_stdout(result)

    status = "valid" if result["valid"] else "invalid"
    area = result["issue_area"] or "unparsed"
    print(
        f"title {status}: type={result['issue_type'] or '-'} area={area} files={len(result['context_files'])}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
