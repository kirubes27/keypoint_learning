"""Quick sanity checks for your models_*.txt files.
- Removes empty lines
- Warns about duplicates
- Writes a cleaned file next to the original
"""
from pathlib import Path
import sys

p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("models_4.txt")
lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip() and not ln.strip().startswith("#")]
dups = sorted({x for x in lines if lines.count(x) > 1})
if dups:
    print("WARNING: duplicates:", dups)
cleaned = p.with_name(p.stem + "_cleaned.txt")
cleaned.write_text("\n".join(lines) + "\n")
print(f"OK: {len(lines)} model names -> wrote {cleaned}")
