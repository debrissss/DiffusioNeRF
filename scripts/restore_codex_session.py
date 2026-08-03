#!/usr/bin/env python3
"""Restore a packaged Codex thread into a local ~/.codex directory."""
import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--codex-home", type=Path, default=Path.home() / ".codex")
    parser.add_argument("--replace-state", action="store_true")
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    target = args.codex_home.expanduser().resolve()
    session_rel = next(bundle.glob("sessions/*/*/*/*.jsonl")).relative_to(bundle)
    target_session = target / session_rel
    target_session.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bundle / session_rel, target_session)
    if args.replace_state:
        state = target / "state_5.sqlite"
        if state.exists():
            shutil.copy2(state, state.with_suffix(".sqlite.before_codex_restore"))
        shutil.copy2(bundle / "state_5.sqlite", state)
    if (bundle / "session_index.jsonl").exists():
        shutil.copy2(bundle / "session_index.jsonl", target / "session_index.jsonl")
    print(f"restored session: {target_session}")
    print("log in to the same ChatGPT account, then restart Codex Desktop")
    if not args.replace_state:
        print("state DB was not replaced; use --replace-state on a blank Codex profile to show the thread in history")


if __name__ == "__main__":
    main()
