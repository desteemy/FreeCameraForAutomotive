# FreeCameraForAutomotive
Vision system designed for automotive to provide virtual camera perspective from four fisheye cameras mounted on vehicle sides.

## Installing required python 3.7 packages using [Anaconda](https://www.anaconda.com/products/individual)
### Using Anaconda Prompt and typing following commands to create a new enviroment
```shell
conda create --name FreeCameraForAutomotive python=3.7
conda install -c conda-forge numpy
conda install -c conda-forge opencv
conda install vispy
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
