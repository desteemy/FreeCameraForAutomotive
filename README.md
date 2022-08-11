# FreeCameraForAutomotive
Vision system designed for automotive to provide virtual camera perspective from four fisheye cameras mounted on vehicle sides.

Sample inputs
<img src="https://user-images.githubusercontent.com/38365841/184258857-01eda3ab-206f-418d-b058-8f0fc452f631.jpg" alt="Front Camera" width="250"/>
<img src="https://user-images.githubusercontent.com/38365841/184258869-28e071ee-2d0f-4426-a6bf-dc8d33d7ca67.jpg" alt="Left Camera" width="250"/>

Sample outputs
<img src="https://user-images.githubusercontent.com/38365841/184258900-a158af23-0e62-4f73-8f17-119daba91ab9.JPG" alt="Sample output" width="250"/>
<img src="https://user-images.githubusercontent.com/38365841/184258905-389a03f2-0ac7-4e01-a120-6192b8c50992.PNG" alt="Sample output" width="250"/>

Keep in mind, that it is not a final product. Some things may be unnecessary and not written in the most optimal form.

## Installing required python 3 packages using [Anaconda](https://www.anaconda.com/products/individual)

### Using Anaconda Prompt and environment.yml

Please edit the environment.yml, so that the last (72) line points to the location of the new enviroment. [More about creating an enviroment from an yml file](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

To use the spec file to create an identical environment:
```shell
conda env create -f environment.yml
```

OR

### Using Anaconda Prompt and requirements.txt on win-64 (May not work on different architecture)

To use the spec file to create an identical environment:
```shell
conda create --name FreeCameraForAutomotive --file requirements.txt
```

To use the spec file to install its listed packages into an existing environment:
```shell
conda install --name FreeCameraForAutomotive --file requirements.txt
```

OR


### Using Anaconda Prompt and typing following commands to create a new enviroment

```shell
conda create --channel conda-forge --name FreeCameraForAutomotive python=3 pillow vispy=0.6 opencv=3.4 pyqt
```
It is possible to swap [PyQt5] for any of following packages: ['PyQt4', 'PyQt5', 'PySide', 'Pyglet', 'Glfw', 'SDL2', 'wx', 'EGL', 'osmesa'].

## Running application
To run simply type `python Video_Viewer.py` in console while in current directory.
Or use Video_Viewer.spec with [PyInstaller]() to create .exe
```shell
conda install -c conda-forge pyinstaller
pyinstaller yourprogram.spec
```
Running last line with additional parameters `--noconfirm` will skip confirming decision and `--onefile` will generate single .exe file instead of whole directory.

By default, the video input is 4 independent .mp4 video files, which can be changed in the [load_cameras] function in the [Video_Processor] class. In other cases, changing the [CAMERA_READ_FROM_FILE] flag to False will capture the video stream.

---
CarModel.obj is a free model with Personal Use License downloaded from [https://free3d.com/3d-model/automobile-v1--84248.html].
And datasets were rendered using [The "Multi-FoV" synthetic datasets](http://rpg.ifi.uzh.ch/fov.html), which was released under the [Creative Commons license (CC BY-NC-SA 3.0)](http://creativecommons.org/licenses/by-nc-sa/3.0/), which is free for non-commercial use (including research).
