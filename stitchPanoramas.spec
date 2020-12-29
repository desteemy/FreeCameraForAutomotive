# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['stitchPanoramas.py'],
             pathex=['C:\\GitFiles\\Python\\InzSystemKamer\\PythonCV2v1'],
             binaries=[('C:\\Users\\Dawid\\Anaconda3\\pkgs\\libopencv-3.4.1-h875b8b8_3\\Library\\bin\\opencv_ffmpeg341_64.dll', '.')],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
a.datas +=Tree(".\\dataset5", prefix="dataset5")
a.datas +=Tree(".\\260-290mp4", prefix="260-290mp4")

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='stitchPanoramas',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='stitchPanoramas')
