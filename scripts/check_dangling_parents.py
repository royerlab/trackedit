#!/usr/bin/env python3
"""Detect the 'dangling parent' corruption in ultrack/trackedit databases.

Invariant that MUST hold in a healthy DB:
    every selected node (selected=1) whose parent_id != -1 must point to a
    parent that also exists AND is selected.

A violation means a selected child references a parent that was deleted
(selected=0) or is missing entirely. This crashes downstream tools that load
the tracking graph. See docs/repair_dangling_parents.md for the fix recipe.

Usage:
    python check_dangling_parents.py /path/to/*.db
    python check_dangling_parents.py data_v1.db data_v2.db
Exit code is non-zero if ANY database is corrupt (usable in CI / batch loops).
"""
import sqlite3
import sys
from glob import glob


def check_db(path):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    cur = con.cursor()
    # selected children whose parent is not a selected node (deleted or missing)
    cur.execute(
        """
        SELECT c.id, c.t, c.parent_id, (p.id IS NULL) AS no_selected_parent
        FROM nodes c
        LEFT JOIN nodes p ON p.id = c.parent_id AND p.selected = 1
        WHERE c.selected = 1 AND c.parent_id != -1 AND p.id IS NULL
        ORDER BY c.t
        """
    )
    rows = cur.fetchall()
    # figure out, per violation, whether the parent row exists at all
    out = []
    for nid, t, pid, _ in rows:
        cur.execute("SELECT selected FROM nodes WHERE id = ?", (pid,))
        prow = cur.fetchone()
        kind = "parent MISSING" if prow is None else "parent selected=0"
        out.append((nid, t, pid, kind))
    con.close()
    return out


def main(patterns):
    paths = []
    for pat in patterns:
        paths.extend(sorted(glob(pat)))
    if not paths:
        print("No database files matched.")
        return 1
    any_bad = False
    for path in paths:
        try:
            v = check_db(path)
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR]    {path}: {e}")
            any_bad = True
            continue
        if v:
            any_bad = True
            print(f"[CORRUPT]  {path}: {len(v)} violation(s)")
            for nid, t, pid, kind in v:
                print(f"    node {nid} (t={t}) -> parent {pid}  [{kind}]")
        else:
            print(f"[OK]       {path}")
    return 1 if any_bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["*.db"]))
