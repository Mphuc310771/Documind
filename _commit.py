import subprocess
import sys

def run(*args: str, allow_fail: bool = False) -> str:
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0 and not allow_fail:
        print(r.stderr or r.stdout, file=sys.stderr)
        sys.exit(r.returncode)
    return (r.stdout or "").strip()

run("git", "-C", r"d:\Code\HKII-2026\SE-TMTriet\doan", "add", "-A")
staged = subprocess.run(
    ["git", "-C", r"d:\Code\HKII-2026\SE-TMTriet\doan", "diff", "--cached", "--quiet"],
    capture_output=True,
)
if staged.returncode == 0:
    print("Nothing to commit")
    sys.exit(0)
parent = run("git", "-C", r"d:\Code\HKII-2026\SE-TMTriet\doan", "rev-parse", "HEAD")
tree = run("git", "-C", r"d:\Code\HKII-2026\SE-TMTriet\doan", "write-tree")
msg = sys.argv[1] if len(sys.argv) > 1 else "Update app"
new = run("git", "-C", r"d:\Code\HKII-2026\SE-TMTriet\doan", "commit-tree", tree, "-p", parent, "-m", msg)
run("git", "-C", r"d:\Code\HKII-2026\SE-TMTriet\doan", "reset", "--hard", new)
print(run("git", "-C", r"d:\Code\HKII-2026\SE-TMTriet\doan", "log", "-1", "--format=full"))
