#!/usr/bin/env python3
"""Package the current Codex thread without copying authentication secrets."""
import hashlib
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

THREAD_ID = "019fa64b-8fdc-79f0-ab96-6fa36accbeda"
CODEX_HOME = Path.home() / ".codex"
STATE_DB = CODEX_HOME / "state_5.sqlite"
SOURCE_SESSION = CODEX_HOME / "sessions/2026/07/28/rollout-2026-07-28T09-16-37-019fa64b-8fdc-79f0-ab96-6fa36accbeda.jsonl"
DEST = Path("/root/autodl-tmp/codex_migration") / f"thread_{THREAD_ID}"
README = """Codex 当前会话迁移包\n\n1. 登录同一个 ChatGPT 账户并关闭 Codex Desktop。\n2. 将整个数据盘保持在 /root/autodl-tmp。\n3. 执行：python /root/autodl-tmp/DiffusioNeRF-main/scripts/restore_codex_session.py . --replace-state\n4. 重启 Codex Desktop。\n\n不包含 auth.json；认证需要在目标机重新登录。\n"""


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if not SOURCE_SESSION.is_file() or not STATE_DB.is_file():
        raise SystemExit("current Codex session or state database is missing")
    DEST.mkdir(parents=True, exist_ok=True)
    rel = Path("sessions/2026/07/28") / SOURCE_SESSION.name
    session_dest = DEST / rel
    session_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_SESSION, session_dest)

    state_dest = DEST / "state_5.sqlite"
    src = sqlite3.connect(str(STATE_DB))
    dst = sqlite3.connect(str(state_dest))
    src.backup(dst)
    dst.close()
    src.close()

    session_index = CODEX_HOME / "session_index.jsonl"
    if session_index.is_file():
        shutil.copy2(session_index, DEST / "session_index.jsonl")

    row = sqlite3.connect(str(STATE_DB)).execute(
        "select id, rollout_path, title, cwd, source, archived, history_mode, updated_at_ms from threads where id=?",
        (THREAD_ID,),
    ).fetchone()
    manifest = {
        "schema": 1,
        "thread_id": THREAD_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_codex_home": str(CODEX_HOME),
        "session_relative_path": str(rel),
        "thread_metadata": row,
        "files": {
            str(rel): {"bytes": session_dest.stat().st_size, "sha256": digest(session_dest)},
            "state_5.sqlite": {"bytes": state_dest.stat().st_size, "sha256": digest(state_dest)},
        },
        "excluded": ["auth.json", "logs_2.sqlite", "memories_1.sqlite", "goals_1.sqlite"],
    }
    if (DEST / "session_index.jsonl").is_file():
        manifest["files"]["session_index.jsonl"] = {
            "bytes": (DEST / "session_index.jsonl").stat().st_size,
            "sha256": digest(DEST / "session_index.jsonl"),
        }
    (DEST / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (DEST / "README.txt").write_text(README)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
