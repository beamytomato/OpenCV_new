# Windows PyInstaller Build

This project can be packaged as a Windows folder-style PyInstaller app using GitHub Actions. The main GUI entrypoint is:

```text
detect_metal_ring_photo.py
```

The combined GUI imports the local live alignment module:

```text
live_reference_alignment.py
```

The Windows app name is:

```text
OpenCVClinicalTools
```

## Build From GitHub Actions

1. Push this repository to GitHub.
2. Open the repository on GitHub.
3. Go to **Actions**.
4. Select **Build Windows PyInstaller App**.
5. Click **Run workflow**.
6. Wait for the `windows-latest` runner to finish.
7. Download the artifact named `OpenCVClinicalTools-windows`.

The workflow builds a folder-style PyInstaller app and zips the full folder:

```text
dist/OpenCVClinicalTools
```

The zip artifact is:

```text
OpenCVClinicalTools-windows.zip
```

## Run On Windows

The Windows user must extract the full zip first.

Do:

```text
Right click OpenCVClinicalTools-windows.zip
Extract All...
Open the extracted OpenCVClinicalTools folder
Double-click OpenCVClinicalTools.exe
```

Important:

- Do not run `OpenCVClinicalTools.exe` directly from the zip preview.
- Do not move `OpenCVClinicalTools.exe` away from its `_internal` folder.
- Keep the full extracted folder together. PyInstaller folder-style apps need the DLLs and Python runtime files inside `_internal`.

If the app is moved without `_internal`, Windows may report missing DLLs such as:

```text
python311.dll was not found
```

## Notes

- The app is intentionally built as a folder-style PyInstaller app, not `--onefile`.
- Folder-style packaging is usually more reliable for OpenCV, NumPy, and Python runtime DLLs.
- The workflow uses Python 3.11 on `windows-latest`.
- The local GUI is served by `detect_metal_ring_photo.py`.
- The app creates or uses relative output folders such as `output/` next to the running app when saving is enabled.
