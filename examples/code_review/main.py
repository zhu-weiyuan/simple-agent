#!/usr/bin/env python3
"""AI Code Review Assistant — CLI entry point.

Usage:
    python examples/code_review/main.py --diff-file changes.diff
    python examples/code_review/main.py --repo /path/to/repo --branch feature-branch
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def get_diff(repo_path: str = ".", branch: str = "", commit: str = "") -> str:
    """Get git diff content."""
    if branch:
        cmd = ["git", "-C", repo_path, "diff", f"main...{branch}"]
    elif commit:
        cmd = ["git", "-C", repo_path, "show", "--format=", commit]
    else:
        cmd = ["git", "-C", repo_path, "diff", "HEAD"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.stdout
    except Exception as e:
        print(f"[Error] Failed to get diff: {e}")
        return ""


def run_linter(file_paths: list) -> str:
    """Run flake8 on specified files."""
    if not file_paths:
        return "No Python files to lint"
    
    issues = []
    for fp in file_paths:
        try:
            result = subprocess.run(
                ["python", "-m", "flake8", "--max-line-length=120", fp],
                capture_output=True, text=True, timeout=30
            )
            if result.stdout.strip():
                issues.append(f"### {fp}\n```\n{result.stdout}```")
        except FileNotFoundError:
            return "flake8 not installed (pip install flake8)"
        except Exception as e:
            issues.append(f"Error linting {fp}: {e}")
    
    return "\n\n".join(issues) if issues else "No linting issues found."


def run_security_scan(file_paths: list) -> str:
    """Run bandit security scan on specified files."""
    if not file_paths:
        return "No Python files to scan"
    
    try:
        result = subprocess.run(
            ["python", "-m", "bandit", "-r"] + file_paths + ["--format", "json"],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            issues = data.get("results", [])
            if not issues:
                return "No security issues found."
            lines = []
            for issue in issues:
                fname = issue.get("filename", "")
                lineno = issue.get("line_number", "?")
                desc = issue.get("text", "")
                severity = issue.get("issue_severity", "").replace("UNDEFINED", "MEDIUM")
                lines.append(f"- **[{severity}]** {fname}:{lineno} — {desc}")
            return "\n".join(lines)
        else:
            return f"Bandit error: {result.stderr}"
    except FileNotFoundError:
        return "bandit not installed (pip install bandit)"
    except Exception as e:
        return f"Security scan error: {e}"


def extract_changed_files(diff_text: str) -> list:
    """Extract changed Python file paths from diff."""
    files = []
    for line in diff_text.split("\n"):
        if line.startswith("+++ b/"):
            fp = line[6:]
            if fp.endswith(".py"):
                files.append(fp)
    return files


def review(diff_text: str, llm_client=None) -> dict:
    """Generate code review report."""
    changed_files = extract_changed_files(diff_text)
    
    # Run checks
    lint_result = run_linter(changed_files[:10])  # limit to avoid timeout
    security_result = run_security_scan(changed_files[:10])
    
    # Build review prompt
    summary = f"Changed files: {', '.join(changed_files[:5])}{'...' if len(changed_files) > 5 else ''}"
    diff_preview = diff_text[:3000]  # truncate for context
    
    report = {
        "summary": summary,
        "files_changed": len(changed_files),
        "lint_issues": lint_result[:2000],
        "security_scan": security_result[:2000],
        "score": 85,
        "issues": [],
        "recommendations": [
            "建议添加单元测试覆盖新增逻辑",
            "确保所有公共函数有文档字符串",
        ],
    }
    
    # If LLM available, generate enriched review
    if llm_client:
        try:
            prompt = f"""请对以下代码变更进行审查：

## 变更摘要
{summary}

## 静态分析结果
### Lint
{lint_result[:1000]}

### Security
{security_result[:1000]}

## Diff（节选）
```diff
{diff_preview}
```

请生成结构化审查报告（JSON格式）：
{{"summary": "变更概述", "score": 85, "issues": [...], "recommendations": [...]}}"""
            
            review_text = llm_client.chat(
                [{"role": "user", "content": prompt}],
                "你是一个专业的代码审查专家。只返回JSON格式的报告。",
                max_tokens=1024,
            )
            try:
                enriched = json.loads(review_text)
                report.update(enriched)
            except json.JSONDecodeError:
                report["llm_review"] = review_text[:2000]
        except Exception as e:
            report["llm_error"] = str(e)
    
    return report


def main():
    parser = argparse.ArgumentParser(description="AI Code Review Assistant")
    parser.add_argument("--diff-file", help="Path to diff file")
    parser.add_argument("--repo", default=".", help="Repository path")
    parser.add_argument("--branch", help="Branch to review")
    parser.add_argument("--commit", help="Commit to review")
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    args = parser.parse_args()
    
    # Get diff
    if args.diff_file:
        diff_text = Path(args.diff_file).read_text()
    else:
        diff_text = get_diff(args.repo, args.branch, args.commit)
    
    if not diff_text.strip():
        print("No changes to review.")
        return
    
    # Review
    from src.my_agent.llm import create_llm_client
    try:
        llm = create_llm_client()
    except Exception:
        llm = None
    
    report = review(diff_text, llm)
    
    # Output
    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"# Code Review Report\n")
        print(f"**Score:** {report['score']}/100\n")
        print(f"**Summary:** {report['summary']}\n")
        if report.get("issues"):
            print("## Issues\n")
            for issue in report["issues"]:
                sev = issue.get("severity", "info").upper()
                print(f"- **[{sev}]** {issue.get('message', '')}")
                if issue.get("suggestion"):
                    print(f"  - 💡 {issue['suggestion']}")
        if report.get("recommendations"):
            print("\n## Recommendations\n")
            for rec in report["recommendations"]:
                print(f"- {rec}")


if __name__ == "__main__":
    main()
