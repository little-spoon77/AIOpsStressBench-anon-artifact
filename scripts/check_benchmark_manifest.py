from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml


INPUT_PATTERN = re.compile(r"\\input\{([^}]+)\}")
INCLUDEGRAPHICS_PATTERN = re.compile(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}")


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def existing(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def check_list(root: Path, paths: list[str], label: str) -> list[str]:
    errors = []
    for raw in paths:
        path = root / raw
        if not existing(path):
            errors.append(f"Missing {label}: {raw}")
    return errors


def check_latex_references(root: Path, paper_path: Path) -> list[str]:
    errors = []
    text = paper_path.read_text(encoding="utf-8")
    paper_dir = paper_path.parent
    for raw in INPUT_PATTERN.findall(text):
        candidate = paper_dir / f"{raw}.tex"
        if not existing(candidate):
            errors.append(f"Missing LaTeX input: {candidate.relative_to(root)}")
    for raw in INCLUDEGRAPHICS_PATTERN.findall(text):
        raw_path = Path(raw)
        candidates = []
        if raw_path.suffix:
            candidates.append(paper_dir / raw_path)
            candidates.append(root / "outputs" / "paper_figures" / raw_path.name)
        else:
            for suffix in [".pdf", ".png"]:
                candidates.append(paper_dir / f"{raw}{suffix}")
                candidates.append(root / "outputs" / "paper_figures" / f"{raw_path.name}{suffix}")
        if not any(existing(path) for path in candidates):
            errors.append(f"Missing figure referenced by LaTeX: {raw}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Check AIOpsStressBench manifest artifacts.")
    parser.add_argument("--manifest", default="benchmark_manifest.yaml")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    manifest_path = root / args.manifest
    manifest = load_manifest(manifest_path)
    errors = []
    errors.extend(check_list(root, [item["file"] for item in manifest.get("datasets", [])], "dataset"))
    errors.extend(check_list(root, manifest.get("required_tables", []), "table"))
    errors.extend(check_list(root, manifest.get("required_figures", []), "figure"))
    primary = manifest.get("primary_outputs", {})
    latex_source = root / primary.get("paper_source", "paper/main.tex")
    if existing(latex_source):
        errors.extend(check_latex_references(root, latex_source))
    else:
        errors.append(f"Missing LaTeX source: {latex_source.relative_to(root)}")
    if errors:
        print("Manifest check failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("Manifest check passed.")


if __name__ == "__main__":
    main()
