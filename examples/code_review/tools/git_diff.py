"""Git diff parsing tool for code review."""
import subprocess
from typing import List, Dict


def parse_diff(diff_text: str) -> Dict:
    """Parse git diff output into structured format."""
    files_changed = []
    current_file = None
    additions = 0
    deletions = 0

    for line in diff_text.split("\n"):
        if line.startswith("diff --git"):
            if current_file:
                files_changed.append(current_file)
            parts = line.split(" b/")
            current_file = {"path": parts[1] if len(parts) > 1 else "", "additions": 0, "deletions": 0}
        elif line.startswith("--- a/") or line.startswith("+++ b/"):
            continue
        elif line.startswith("@@"):
            continue
        elif line.startswith("+") and current_file:
            current_file["additions"] += 1
        elif line.startswith("-") and current_file:
            current_file["deletions"] += 1

    if current_file:
        files_changed.append(current_file)

    total_add = sum(f["additions"] for f in files_changed)
    total_del = sum(f["deletions"] for f in files_changed)

    return {
        "files": files_changed,
        "total_files": len(files_changed),
        "total_additions": total_add,
        "total_deletions": total_del,
    }


def get_git_diff(repo_path: str = ".", ref: str = "HEAD") -> str:
    """Get git diff from repository."""
    try:
        result = subprocess.run(
            ["git", "-C", repo_path, "diff", ref],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout
    except Exception as e:
        return f"Error getting diff: {e}"
