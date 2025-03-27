[![PyPI - Version](https://img.shields.io/pypi/v/trackedit.svg)](https://pypi.org/project/trackedit)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/trackedit.svg)](https://pypi.org/project/trackedit)
[![codecov](https://codecov.io/gh/royerlab/trackedit/branch/main/graph/badge.svg)](https://codecov.io/gh/royerlab/trackedit)

# TrackEdit ⛓️

Set of napari widget to interactively proofread, edit, and annotate, cell tracking data tracked with [Ultrack](https://github.com/royerlab/ultrack).

-----

`TrackEdit` napari UI:

<img width="1449" alt="UI2" src="https://github.com/user-attachments/assets/c9d0c209-cb87-4820-af68-1744ef4dcb90" />

## Installation HPC 🖥️

1. Clone repo in $MYDATA and install pixi environment
```console
git clone https://github.com/royerlab/trackedit.git --recurse-submodules
cd trackedit
module load pixi
pixi install
pixi shell
```
2. `cd` to relevant data folder (that contains `data.db` and `metadata.toml`), copy `scripts/demo.py` script, change relevant input parameters, and run inTRACKtive:
```
python demo_neuromast.py #_after changing relevant parameters
```
Note: make sure to first `pixi shell` in the $MYDATA directory, before running the python script. This is the equivalent of `conda activate ...`

### Keyboard shortcuts ⌨️

#### Napari Viewer and Layer Controls
| Mouse / Key binding | Action |
|-------------------|---------|
| Click on point/label | Select node (centers view if needed) |
| SHIFT + click | Add node to selection |
| Q | Toggle between all nodes view and selected lineages only |

#### Tree View Controls
| Mouse / Key binding | Action |
|-------------------|---------|
| Click on node | Select node (centers view if needed) |
| SHIFT + click | Add node to selection |
| Scroll | Zoom in/out |
| Scroll + X / Right click + drag horizontally | Zoom x-axis only |
| Scroll + Y / Right click + drag vertically | Zoom y-axis only |
| Mouse drag | Pan |
| SHIFT + Mouse drag | Rectangular node selection |
| Right click | Reset view |
| Q | Toggle between all lineages (vertical) and selected lineages (horizontal) |
| ~~W~~ | ~~Switch between lineage tree and object size plot~~ |
| ← | Select node to the left |
| → | Select node to the right |
| ↑ | Select parent node (vertical) or next lineage (horizontal) |
| ↓ | Select child node (vertical) or previous lineage (horizontal) |

#### Track Editing
| Mouse / Key binding | Action |
|-------------------|---------|
| D | Delete selected nodes |
| B | Break edge between selected nodes |
| A | Create edge between selected nodes (if valid) |
| Z | Undo last edit |
| R | Redo last edit |

## Credits 🙌
The `TrackEdit` widget structure is inspired by, and built upon, [Motile_tracker](https://github.com/funkelab/motile_tracker) (Funke lab, HHMI Janelia Research Campus)

## License 🔓

`trackedit` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
