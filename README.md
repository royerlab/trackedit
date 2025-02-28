[![PyPI - Version](https://img.shields.io/pypi/v/trackedit.svg)](https://pypi.org/project/trackedit)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/trackedit.svg)](https://pypi.org/project/trackedit)

# TrackEdit

Set of napari widget to interactively proofread, edit, and annotate, cell tracking data tracked with [Ultrack](https://github.com/royerlab/ultrack). 

-----

`TrackEdit` napari UI:

<img width="1449" alt="UI2" src="https://github.com/user-attachments/assets/c9d0c209-cb87-4820-af68-1744ef4dcb90" />

## Installation HPC

```console
git clone https://github.com/royerlab/trackedit.git
cd trackedit
module load pixi
pixi install
pixi shell
python scripts/demo_neuromast.py  #after changing working directory path in this file
```

## Credits
The `TrackEdit` widget structure is inspired by, and built upon, [Motile_tracker](https://github.com/funkelab/motile_tracker) (Funke lab, HHMI Janelia Research Campus)

## License

`trackedit` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

