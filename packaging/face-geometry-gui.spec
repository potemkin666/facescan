# PyInstaller spec that packages the "Moonlit Shore" desktop GUI into a
# standalone, double-clickable application (no terminal, no Python install
# required by the end user).
#
# Build (from the repository root, with `pip install -e '.[app]'` first):
#
#     pyinstaller packaging/face-geometry-gui.spec
#
# The finished app is written to dist/:
#   * macOS   -> dist/Face Geometry.app   (double-click to launch)
#   * Windows -> dist/Face Geometry/Face Geometry.exe
#   * Linux   -> dist/Face Geometry/Face Geometry (executable)

# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ["../face_geometry/gui.py"],
    pathex=["../"],
    binaries=[],
    datas=[],
    hiddenimports=["face_geometry"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="Face Geometry",
    debug=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=True,
)

app = BUNDLE(
    exe,
    name="Face Geometry.app",
    icon=None,
    bundle_identifier="com.facescan.facegeometry",
    info_plist={
        "CFBundleName": "Face Geometry",
        "CFBundleDisplayName": "Face Geometry",
        "CFBundleShortVersionString": "0.1.0",
        "NSHighResolutionCapable": True,
    },
)
