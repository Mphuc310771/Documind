"""Stage and commit without Cursor Co-authored-by trailer."""
import subprocess
import sys

def run(*args: str, check: bool = True) -> str:
    r = subprocess.run(args, capture_output=True, text=True)
    if check and r.returncode != 0:
        print(r.stderr or r.stdout, file=sys.stderr)
        sys.exit(r.returncode)
    return (r.stdout or "").strip()

run("git", "add", "-A")
if not run("git", "diff", "--cached", "--quiet", check=False):
    tree = run("git", "write-tree")
    parent = run("git", "rev-parse", "HEAD")
    msg = sys.argv[1] if len(sys.argv) > 1 else "Cleanup"
    new = run("git", "commit-tree", tree, "-p", parent, "-m", msg)
    run("git", "reset", "--hard", new)
    print(run("git", "log", "-1", "--oneline"))
else:
    print("Nothing to commit")
