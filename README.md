# FreeCameraForAutomotive
Vision system designed for automotive to provide virtual camera perspective from four fisheye cameras mounted on vehicle sides.

## Installing required python 3.6 packages using [Anaconda](https://www.anaconda.com/products/individual)
### Using Anaconda Prompt and typing following commands to create a new enviroment
(opencv in version >=3.3 is necessary; 3.3 version guarantees there will be no problem with importing cv2 dll, which can occur in newer versions, and python 3.6 is the newest for that opencv version).
```shell
conda create --name FreeCameraForAutomotive python=3.6
conda activate FreeCameraForAutomotive
conda install -c conda-forge numpy
conda install -c conda-forge ffmpeg
conda install -c conda-forge opencv=3.3
conda install vispy
```
and any of following packages: ['PyQt4', 'PyQt5', 'PySide', 'Pyglet', 'Glfw', 'SDL2', 'wx', 'EGL', 'osmesa']. We recommend PyQt
```shell
conda install -c conda-forge pyqt
```

OR

### Using Anaconda Prompt and spec-file.txt on win-64

To use the spec file to create an identical environment:
```shell
conda create --name myenv --file spec-file.txt
```

To use the spec file to install its listed packages into an existing environment:
```shell
conda install --name myenv --file spec-file.txt
```

## Running application
To run simply type `python Video_Viewer.py` in console while in current directory.
Or use Video_Viewer.spec with [PyInstaller]() to create .exe
```shell
conda install -c conda-forge pyinstaller
pyinstaller yourprogram.spec
```
Running last line with additional parameters `--noconfirm` will skip confirming decision and `--onefile` will generate single .exe file instead of whole directory.



---
CarModel.obj is a free model with Personal Use License downloaded from [https://free3d.com/3d-model/automobile-v1--84248.html].
And datasets were rendered using [The "Multi-FoV" synthetic datasets](http://rpg.ifi.uzh.ch/fov.html), which was released under the [Creative Commons license (CC BY-NC-SA 3.0)](http://creativecommons.org/licenses/by-nc-sa/3.0/), which is free for non-commercial use (including research).
