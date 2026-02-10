#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def load_patch_module(patch_dir: Path):
    patch_file = patch_dir / "patch.py"
    if not patch_file.exists():
        raise FileNotFoundError(f"missing patch.py in {patch_dir}")
    module_name = f"demo_patch_{patch_dir.name}"
    spec = importlib.util.spec_from_file_location(module_name, patch_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module from {patch_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_patch_name(patch_dir: Path) -> str:
    metadata_file = patch_dir / "patch.json"
    if not metadata_file.exists():
        return patch_dir.name
    try:
        metadata = json.loads(metadata_file.read_text())
        return str(metadata.get("name", patch_dir.name))
    except Exception:
        return patch_dir.name


def discover_patches(patches_root: Path) -> list[Path]:
    if not patches_root.exists():
        return []
    return sorted(
        entry
        for entry in patches_root.iterdir()
        if entry.is_dir() and (entry / "patch.py").exists()
    )


def run_verify_script(script_path: Path, repo_root: Path) -> tuple[bool, str]:
    completed = subprocess.run(["bash", str(script_path)], cwd=repo_root)
    if completed.returncode != 0:
        return False, f"{script_path} exited {completed.returncode}"
    return True, f"{script_path} passed"


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply demo patch queue")
    parser.add_argument("--dry-run", action="store_true", help="do not modify files")
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="skip verify() / verify.sh execution",
    )
    args = parser.parse_args()

    this_file = Path(__file__).resolve()
    queue_root = this_file.parent
    repo_root = queue_root.parent
    patches_root = queue_root / "patches"
    patch_dirs = discover_patches(patches_root)

    if not patch_dirs:
        print("No patches found.")
        return 0

    applied_count = 0
    skipped_count = 0
    verified_count = 0

    for patch_dir in patch_dirs:
        patch_name = read_patch_name(patch_dir)
        module = load_patch_module(patch_dir)
        patch_id = str(getattr(module, "PATCH_ID", patch_dir.name))

        print(f"[PATCH] {patch_id} - {patch_name}")
        detect_result = module.detect(repo_root)
        already_applied = bool(detect_result.get("applied", False))

        if already_applied:
            print("  SKIP: already applied")
            skipped_count += 1
        else:
            if args.dry_run:
                print("  DRY-RUN: would apply")
            else:
                apply_result = module.apply(repo_root, dry_run=False)
                if not apply_result.get("changed", False):
                    print("  SKIP: no file changes produced")
                    skipped_count += 1
                else:
                    change_list = apply_result.get("changes", [])
                    joined_changes = ", ".join(change_list) if change_list else "updated"
                    print(f"  APPLY: {joined_changes}")
                    applied_count += 1

                post_detect = module.detect(repo_root)
                if not bool(post_detect.get("applied", False)):
                    raise RuntimeError(
                        f"{patch_id}: post-apply detect indicates patch is incomplete"
                    )

        if args.dry_run or args.skip_verify:
            continue

        verified = False
        if hasattr(module, "verify"):
            ok, message = module.verify(repo_root)
            if not ok:
                raise RuntimeError(f"{patch_id}: verify failed: {message}")
            print(f"  VERIFY: {message}")
            verified = True
        else:
            verify_script = patch_dir / "verify.sh"
            if verify_script.exists():
                ok, message = run_verify_script(verify_script, repo_root)
                if not ok:
                    raise RuntimeError(f"{patch_id}: verify failed: {message}")
                print(f"  VERIFY: {message}")
                verified = True

        if verified:
            verified_count += 1

    print(
        f"Summary: applied={applied_count} skipped={skipped_count} "
        f"verified={verified_count} dry_run={args.dry_run}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
