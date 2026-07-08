# Daily Development Log

Project: OpenCV metal ring detection, skin landmark labeling, and live reference alignment  
Primary files: `detect_metal_ring_photo.py`, `live_reference_alignment.py`  
Log scope: work beginning with the first browser user interface for `detect_metal_ring_photo.py`

## 2026-06-30

### Goals

- Create a browser-based graphical user interface for `detect_metal_ring_photo.py`.
- Keep the existing OpenCV detection logic intact while making the tool easier to use.
- Support image upload, browser camera capture, automatic metal ring detection, automatic skin landmark detection, manual skin landmark labeling, and output preview.
- Avoid unnecessary disk usage from repeated debug image generation.
- Combine the metal ring/landmark tool and live reference alignment tool into one local browser interface.

### Work Completed

#### Built a localhost GUI inside `detect_metal_ring_photo.py`

Implemented a built-in `http.server` interface directly in the existing detector file. The GUI is launched with:

```bash
python3 detect_metal_ring_photo.py --gui
```

Key server entrypoint:

```python
def run_gui_server(host="127.0.0.1", port=8501, alignment_reference=None, camera_index=0):
    """Start the built-in local browser GUI."""
    alignment_args = create_alignment_args(
        host,
        port,
        reference_image=alignment_reference,
        camera_index=camera_index,
    )
    alignment_state = alignment_gui.AlignmentWebState(alignment_args)
```

The GUI was embedded as HTML/JavaScript strings and served from the same Python file:

```python
GUI_PAGE = r"""<!doctype html>
<html lang="en">
...
</html>"""
```

#### Added image upload and browser camera capture

The GUI accepts an uploaded image or an image captured from the browser camera. Captured/uploaded images are sent to `/api/analyze`, decoded by OpenCV, and passed into the existing ring and landmark functions.

Relevant API route:

```python
if parsed.path == "/api/analyze":
    payload = self.read_analysis_request()
    result = process_gui_image(**payload)
    response = detection_to_gui_summary(result, payload["landmark_mode"])
    self.send_json(response)
```

#### Added automatic/manual landmark modes

The GUI supports:

- automatic skin landmarks
- manual skin landmarks
- ring-only mode

Manual GUI boxes are converted into the same landmark structure used by automatic detections:

```python
def landmarks_from_gui_boxes(boxes):
    """Convert browser-drawn box dictionaries into landmark records."""
    landmarks = []
    for box in boxes:
        landmark = make_manual_landmark(
            float(box.get("x", 0)),
            float(box.get("y", 0)),
            float(box.get("x", 0)) + float(box.get("width", 0)),
            float(box.get("y", 0)) + float(box.get("height", 0)),
        )
        if landmark is not None:
            landmarks.append(landmark)
    return landmarks
```

#### Added manual landmark coordinate JSON export

When output saving is enabled and manual landmarks are used, the manually drawn boxes can be saved as JSON:

```python
def save_manual_landmarks(image_path, landmarks):
    """Save manually drawn landmarks as JSON next to the annotated output."""
    image_path = Path(image_path)
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{image_path.stem}_manual_landmarks.json"
```

#### Added option to skip ring detection

The command-line and GUI now support skipping metal ring detection:

```bash
python3 detect_metal_ring_photo.py image.jpg --no-ring
```

Core implementation:

```python
if args.no_ring:
    detection = None
    debug = {}
    annotated = image.copy()
else:
    detection, debug = detect_ring(image)
    annotated = annotate_ring(image, detection, show_raw_contours=False)
```

The GUI equivalent uses:

```python
ring_skipped = ring_mode == "skip"
```

#### Changed saving behavior to default off

Originally every run saved annotated images and all debug masks. This generated many files in `output/`. The default was changed so output images are not saved unless requested.

New CLI opt-in:

```bash
python3 detect_metal_ring_photo.py image.jpg --save-images
```

Implementation:

```python
def save_outputs(image_path, annotated, debug, save_images=False):
    """Save the annotated image and debug images."""
    if not save_images:
        return None, {}
```

The GUI now has an unchecked-by-default **Save output images** checkbox.

#### Added in-memory GUI preview when not saving

Because saving is now default off, the browser still needs to show the annotated image without writing it to disk. Added image-to-data-URL encoding:

```python
def image_to_data_url(image, extension=".jpg"):
    """Encode an OpenCV image as a browser-displayable data URL."""
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise RuntimeError("Could not encode annotated image for preview.")
    mime_type = "image/png" if extension.lower() == ".png" else "image/jpeg"
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"
```

#### Reduced obstruction from "Metal ring not detected" text

The failure label was too large and covered useful image content. It was moved into the corner and reduced in size:

```python
cv2.putText(
    output,
    "Metal ring not detected",
    (8, 18),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.45,
    (0, 0, 255),
    1,
    cv2.LINE_AA,
)
```

#### Combined two GUIs into one local interface

The metal ring/landmark GUI and the live reference alignment GUI were combined behind one local server with two tabs:

- Metal Ring and Landmarks
- Live Reference Alignment

Combined shell:

```python
COMBINED_GUI_PAGE = r"""<!doctype html>
...
<button id="metalTab" class="active" type="button">Metal Ring and Landmarks</button>
<button id="alignmentTab" type="button">Live Reference Alignment</button>
...
<iframe id="metalFrame" class="active" src="/metal"></iframe>
<iframe id="alignmentFrame" src="/alignment"></iframe>
...
</html>"""
```

Routes:

```python
if parsed.path == "/":
    self.send_html(COMBINED_GUI_PAGE)
elif parsed.path == "/metal":
    self.send_html(GUI_PAGE)
elif parsed.path == "/alignment":
    self.send_html(alignment_gui.WEB_APP_HTML)
```

The live alignment page reuses the existing state object from `live_reference_alignment.py`:

```python
alignment_state = alignment_gui.AlignmentWebState(alignment_args)
server = CombinedGuiServer((host, port), MetalRingGuiHandler, alignment_state)
```

Optional preload:

```bash
python3 detect_metal_ring_photo.py --gui --alignment-reference path/to/reference.jpg
```

### Challenges and Resolutions

#### Port already in use

Problem: running `python3 detect_metal_ring_photo.py --gui` produced:

```text
OSError: [Errno 48] Address already in use
```

Cause: an earlier GUI server was already running on `127.0.0.1:8501`.

Resolution:

- Stop the prior server with `Control+C`.
- Add clearer error handling so future failures explain what happened.

Implemented message:

```python
if exc.errno == errno.EADDRINUSE:
    url = f"http://{host}:{port}"
    raise SystemExit(
        f"Port {port} is already in use.\n"
        f"If the GUI is already open, use {url}.\n"
        "Otherwise stop the earlier server with Control+C in its terminal, "
        f"or start a second copy with: python3 detect_metal_ring_photo.py --gui --port {port + 1}"
    ) from exc
```

#### Manual box cursor offset

Problem: manual landmark boxes appeared offset from the cursor location in the browser.

Cause: JavaScript used the canvas internal pixel dimensions instead of the actual rendered browser rectangle.

Resolution:

```javascript
function canvasEventToImagePoint(e) {
  const r = manualCanvas.getBoundingClientRect();
  const x = (e.clientX - r.left) * imageBitmap.width / r.width;
  const y = (e.clientY - r.top) * imageBitmap.height / r.height;
  return {
    x: Math.max(0, Math.min(imageBitmap.width, x)),
    y: Math.max(0, Math.min(imageBitmap.height, y))
  };
}
```

#### Too many output files

Problem: every run saved the annotated output plus many debug masks.

Resolution:

- Default saving changed to off.
- CLI added `--save-images`.
- GUI added an unchecked-by-default checkbox.
- GUI uses data URLs for previews when not saving.

#### Combining two independently served GUIs

Problem: both tools had their own browser interfaces and route structures.

Resolution:

- `detect_metal_ring_photo.py` became the combined launcher.
- `/metal` serves the metal ring/landmark GUI.
- `/alignment` serves the existing alignment GUI HTML from `live_reference_alignment.py`.
- `/api/analyze` remains dedicated to metal ring/landmark analysis.
- `/api/state`, `/api/control`, `/api/start`, `/api/stop`, `/api/reference`, `/stream`, and related routes are delegated to the live alignment state machine.

### Verification

Commands run during development:

```bash
python3 -c "compile(open('detect_metal_ring_photo.py').read(), 'detect_metal_ring_photo.py', 'exec'); print('syntax ok')"
python3 detect_metal_ring_photo.py --help
python3 detect_metal_ring_photo.py image.jpg --no-ring
python3 detect_metal_ring_photo.py image.jpg --save-images
python3 detect_metal_ring_photo.py --gui --port 8503
```

Verified routes on the combined GUI:

```text
GET /           -> 200 text/html
GET /metal      -> 200 text/html
GET /alignment  -> 200 text/html
GET /api/state  -> 200 application/json
```

### Current Usage

Start combined GUI:

```bash
python3 detect_metal_ring_photo.py --gui
```

Open:

```text
http://127.0.0.1:8501
```

Start GUI with preloaded alignment reference:

```bash
python3 detect_metal_ring_photo.py --gui --alignment-reference path/to/reference.jpg
```

CLI, no saving by default:

```bash
python3 detect_metal_ring_photo.py image.jpg
```

CLI, save outputs:

```bash
python3 detect_metal_ring_photo.py image.jpg --save-images
```

CLI, skip ring detection:

```bash
python3 detect_metal_ring_photo.py image.jpg --no-ring
```

### Notes for Future Work

- Consider separating the large embedded HTML strings into template files once the interface stabilizes.
- Consider adding a run-history table in the GUI if saved outputs become useful again.
- Consider adding a small status badge to show whether output saving is enabled.
- Consider adding browser-side keyboard shortcuts for manual landmark labeling.
- Consider adding per-run metadata JSON for both tools when saving is enabled.

## 2026-07-01

### Goals

- Document the work completed since the first `detect_metal_ring_photo.py` GUI was created.
- Capture implementation decisions, useful code snippets, encountered issues, and current usage.

### Work Completed

- Created this professional daily development log.
- Organized the project history into goals, completed work, code snippets, challenges, verification, and future work.
- Anchored the log to the current implementation in `detect_metal_ring_photo.py` and `live_reference_alignment.py`.

### Current Status

- The combined localhost GUI is implemented in `detect_metal_ring_photo.py`.
- The live reference alignment tab reuses the existing alignment logic from `live_reference_alignment.py`.
- Output image saving is default off.
- The manual landmark cursor offset issue has been fixed.
- The project is ready for continued testing with real patient/skin images and live webcam workflows.
