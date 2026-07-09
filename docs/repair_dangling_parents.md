# Recipe: rescuing a corrupt TrackEdit database ("dangling parent")

This is the exact procedure used to rescue 3 databases (neuromast2_t154 v18→v19,
neuromast3_t308 v1→v2, neuromast2_t574 v2→v3). Follow it whenever a downstream
tool crashes loading a curated `data_vN.db`.

---

## The invariant that was broken

> Every **selected** node (`selected=1`) whose `parent_id != -1` must point to a
> parent that **exists** and is **selected**.

A violation = a selected child pointing to a parent that was deleted
(`selected=0`) or is missing. Downstream graph loaders choke on it.

## Why it happens (root cause)

When you delete a node, TrackEdit's orphan-cleanup only resets the `parent_id`
of children that are inside the **loaded window** (`selected` and `t < Tmax`).
If a child sits just outside that window, its back-reference is never cleared,
so it dangles. Two ways a child ends up outside the window:

1. **Auto-shrink off-by-one** (now fixed in `DatabaseHandler.check_if_tmax_changed`):
   `Tmax` was set to `current_max_time` instead of `current_max_time + 1`. Since
   `Tmax` is *exclusive*, the genuine last populated frame was evicted every
   session. Deleting a node at the new "last" frame orphaned its child one frame
   beyond. → e.g. neuromast2_t154 (t=152), neuromast3_t308 (t=306).
2. **Deliberate small `tmax`** (inherent, NOT fixed by code): you opened with
   `tmax=T` smaller than the data extent and curated `t=0..T-1`. Deleting a node
   at `t=T-1` orphans its child at `t=T`, which was never loaded. → neuromast2_t574
   (`tmax=550`, child at t=550). The protection here is recording `Tmax` in the
   changelog so downstream loads only the curated window.

The code fix stops (1) from creating *new* corruption. Existing corrupt DBs must
be repaired manually with this recipe. (2) can still occur by design — mitigate
by keeping the `Tmax` line accurate and having downstream respect it.

---

## Procedure

### 0. Safety rules
- **Never edit an existing `data_vN.db` in place.** Always copy to the next
  version number and edit the copy.
- Leave the original `data.db` and all prior versions untouched (they are the
  record). `cp -n` (no-clobber) so you never overwrite an existing version.

### 1. Detect
```bash
python scripts/check_dangling_parents.py "/path/to/.../tracked/data_v*.db"
```
Also check the original to confirm it is clean (corruption came from editing):
```bash
python scripts/check_dangling_parents.py "/path/to/.../tracked/data.db"
```
Note the **latest** version present — that is what you branch the fix from.

### 2. Classify each violation
For each corrupt child `C` (`selected=1`, `parent_id=P`) run:
```python
import sqlite3
con = sqlite3.connect("file:DB?mode=ro", uri=True); cur = con.cursor()
cur.execute("SELECT id,t,parent_id,selected FROM nodes WHERE id=?", (C,)); print("child :", cur.fetchone())
cur.execute("SELECT id,t,parent_id,selected FROM nodes WHERE id=?", (P,)); print("parent:", cur.fetchone())
cur.execute("SELECT id,t,selected FROM nodes WHERE parent_id=? AND selected=1", (C,)); print("kids of child:", cur.fetchall())
cur.execute("SELECT max(t) FROM nodes WHERE selected=1"); print("max selected t:", cur.fetchone()[0])
```
Determine:
- **Is C a leaf** (no selected children) or does it **anchor a downstream track**?
- Is C at the **true last frame** (case 1) or **just past a deliberate tmax** (case 2)?

### 3. Decide the repair per node
| Situation | Repair | SQL |
|---|---|---|
| C is a **leaf** and the track tail was meant to be trimmed | **DELETE** | `selected=0, parent_id=-1` |
| C **anchors a downstream track** (has descendants) | **REPARENT** (keep the track) | `parent_id=-1` (leave `selected=1`) |
| Unsure | **REPARENT** (non-destructive default) | `parent_id=-1` |

Deleting a node that has descendants would orphan them (re-creating the bug) or
force cascade-deleting a real track — so only delete leaves.

### 4. Apply to a new version
```bash
LATEST=data_v1.db          # <- the highest existing version
NEW=data_v2.db             # <- LATEST version number + 1
cp -n "$LATEST" "$NEW"

python3 - "$NEW" <<'PY'
import sqlite3, sys
con = sqlite3.connect(sys.argv[1]); cur = con.cursor()
# --- edit these per your decision in step 3 ---
cur.execute("UPDATE nodes SET selected=0, parent_id=-1 WHERE id=?", (307000039,))  # DELETE a leaf
# cur.execute("UPDATE nodes SET parent_id=-1 WHERE id=?", (551000029,))            # REPARENT (keep selected)
con.commit()
# verify
cur.execute("""SELECT c.id,c.t,c.parent_id FROM nodes c
   LEFT JOIN nodes p ON p.id=c.parent_id AND p.selected=1
   WHERE c.selected=1 AND c.parent_id!=-1 AND p.id IS NULL""")
print("remaining violations:", cur.fetchall())
con.close()
PY
```
Expect `remaining violations: []`. Re-run the checker on `$NEW` to be sure.

### 5. Determine the correct `Tmax` for the changelog
This is the frame bound downstream should load. Compute what a **fixed** TrackEdit
would log:
- **Deliberate small tmax** (case 2): use the run script's `tmax` (the curation
  window). e.g. neuromast2_t574 → `Tmax: 550`.
- **Full/auto (case 1)**: `Tmax = min(script_tmax clamped to array frames,
  max_selected_t + 1)`. e.g. neuromast3_t308 → `min(600→308, 306+1) = 307`.

### 6. Write `data_v{N+1}_changelog.txt`
Match the normal TrackEdit format and **always include the `Parameters: Tmax`
line** (downstream relies on it). Entries are tab-indented. End with a `#` note
block documenting the manual repair. Template:
```
Start annotation session - TrackEdit vX.Y.Z (YYYY-MM-DD HH:MM:SS)
Parameters: Tmax: <T>, working_directory: ., db_filename: data_v{N+1}.db
DeleteEdges:[(<parent>, <child>)]
		[YYYY-MM-DD HH:MM:SS] db: setting parent_id[id=<child>] = -1 (was <parent>)
DeleteNodes:[<child>]                 # ONLY for the DELETE case, omit for REPARENT
		[YYYY-MM-DD HH:MM:SS] db: setting selected[id=<child>] = 0 (was True)

# ----------------------------------------------------------------------------
# NOTE: data_v{N+1}.db is a manual repair of data_v{N}.db (not a live session).
# <one line: which node, which deleted parent, from which prior changelog>
# Cause: <case 1 off-by-one  |  case 2 deliberate tmax=T boundary>.
# Repair: <DELETE leaf | REPARENT track-anchor>.
# Verification: full re-scan -> 0 remaining violations.
# ----------------------------------------------------------------------------
```

### 7. Downstream
- Any exported artifacts (`*_tracks.csv`, `*_segments.zarr`, `*_annotations.zarr`)
  predate the repair — **regenerate them from the new version**.
- Confirm downstream honors the `Tmax` bound so it does not load un-curated
  frames beyond the curation window.

---

## Prevention
- Keep TrackEdit updated (the `check_if_tmax_changed` off-by-one is fixed).
- Periodically sweep with `scripts/check_dangling_parents.py` across dataset
  folders — this corruption was found in multiple experiments.
