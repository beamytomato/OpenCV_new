import argparse
import json
import math
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
from local_form import FieldStorage
import numpy as np

# Match-quality gates for deciding whether guidance is trustworthy.
MIN_INLIERS_FOR_GUIDANCE = 10
MIN_INLIER_RATIO = 0.30
MIN_VALID_SCALE = 0.60
MAX_VALID_SCALE = 1.60
MAX_VALID_ROTATION = 25.0
MAX_VALID_TRANSLATION = 350.0


WINDOW_NAME = "Live Reference Alignment"
DRAW_MASK_WINDOW = "Draw Reference Mask"
ALPHA_TRACKBAR = "Overlay opacity"
SCALE_TRACKBAR = "Live scale"
X_TRACKBAR = "Live X offset"
Y_TRACKBAR = "Live Y offset"
DEFAULT_MAX_MATCHES = 40
DEFAULT_BRUSH_RADIUS = 18
MIN_BRUSH_RADIUS = 3
MAX_BRUSH_RADIUS = 80
DEFAULT_MAX_MARKERS = 8
MAX_MARKER_HYPOTHESIS_POINTS = 5
DEFAULT_MARKER_DETECT_SCALE = 0.4
MIN_MARKER_INLIERS_FOR_GUIDANCE = 3
MIN_MARKER_INLIER_RATIO = 0.45
DEFAULT_VISUAL_SCORE_SCALE = 0.35
VISUAL_NUDGE_PIXELS = 18


def nothing(_value):
    """OpenCV trackbars require a callback function."""
    pass


def load_reference_image(path):
    """Load the saved reference image from disk."""
    reference_path = Path(path)

    if not reference_path.exists():
        raise FileNotFoundError(f"Reference image does not exist: {reference_path}")

    reference = cv2.imread(str(reference_path))

    if reference is None:
        raise ValueError(f"OpenCV could not load reference image: {reference_path}")

    return reference


def resize_reference_for_preview(reference, preview_width):
    """Resize a large reference image so the live UI stays responsive."""
    height, width = reference.shape[:2]

    if width <= preview_width:
        return reference.copy()

    scale = preview_width / width
    preview_height = int(height * scale)

    return cv2.resize(
        reference,
        (preview_width, preview_height),
        interpolation=cv2.INTER_AREA,
    )


def resize_with_aspect_ratio(image, target_width, target_height, fit_mode):
    """
    Resize an image without stretching it.

    contain: show the whole image, with possible empty margins.
    cover: fill the target area, with possible cropping.
    """
    image_height, image_width = image.shape[:2]
    width_scale = target_width / image_width
    height_scale = target_height / image_height

    if fit_mode == "cover":
        scale = max(width_scale, height_scale)
    else:
        scale = min(width_scale, height_scale)

    new_width = max(1, int(image_width * scale))
    new_height = max(1, int(image_height * scale))

    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)


def center_on_canvas(image, canvas_shape):
    """Place an image in the center of a black canvas."""
    canvas_height, canvas_width = canvas_shape[:2]
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    image_height, image_width = image.shape[:2]
    x_start = (canvas_width - image_width) // 2
    y_start = (canvas_height - image_height) // 2
    x_end = x_start + image_width
    y_end = y_start + image_height

    canvas_x1 = max(0, x_start)
    canvas_y1 = max(0, y_start)
    canvas_x2 = min(canvas_width, x_end)
    canvas_y2 = min(canvas_height, y_end)

    image_x1 = max(0, -x_start)
    image_y1 = max(0, -y_start)
    image_x2 = image_x1 + (canvas_x2 - canvas_x1)
    image_y2 = image_y1 + (canvas_y2 - canvas_y1)

    if canvas_x2 > canvas_x1 and canvas_y2 > canvas_y1:
        canvas[canvas_y1:canvas_y2, canvas_x1:canvas_x2] = image[
            image_y1:image_y2,
            image_x1:image_x2,
        ]

    return canvas


def prepare_side_by_side_live_frame(frame, target_shape):
    """Resize the live frame to match the reference preview canvas."""
    target_height, target_width = target_shape[:2]
    resized = resize_with_aspect_ratio(frame, target_width, target_height, "contain")
    return center_on_canvas(resized, target_shape)


def scale_live_for_overlay(frame, target_width, target_height, scale_percent, fit_mode):
    """Resize the live frame for overlay mode without distorting it."""
    fitted = resize_with_aspect_ratio(frame, target_width, target_height, fit_mode)

    slider_scale = max(10, scale_percent) / 100.0
    new_width = max(1, int(fitted.shape[1] * slider_scale))
    new_height = max(1, int(fitted.shape[0] * slider_scale))

    return cv2.resize(fitted, (new_width, new_height), interpolation=cv2.INTER_LINEAR)


def place_live_on_reference_canvas(live_frame, reference_shape, x_offset, y_offset):
    """Place the live frame on a reference-sized canvas and return a mask."""
    canvas_height, canvas_width = reference_shape[:2]
    live_height, live_width = live_frame.shape[:2]

    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

    x_start = ((canvas_width - live_width) // 2) + x_offset
    y_start = ((canvas_height - live_height) // 2) + y_offset
    x_end = x_start + live_width
    y_end = y_start + live_height

    canvas_x1 = max(0, x_start)
    canvas_y1 = max(0, y_start)
    canvas_x2 = min(canvas_width, x_end)
    canvas_y2 = min(canvas_height, y_end)

    live_x1 = max(0, -x_start)
    live_y1 = max(0, -y_start)
    live_x2 = live_x1 + (canvas_x2 - canvas_x1)
    live_y2 = live_y1 + (canvas_y2 - canvas_y1)

    if canvas_x2 > canvas_x1 and canvas_y2 > canvas_y1:
        canvas[canvas_y1:canvas_y2, canvas_x1:canvas_x2] = live_frame[
            live_y1:live_y2,
            live_x1:live_x2,
        ]
        mask[canvas_y1:canvas_y2, canvas_x1:canvas_x2] = 255

    return canvas, mask


def blend_overlay(reference, live_canvas, live_mask, alpha):
    """Blend the live frame over the reference only where live pixels exist."""
    blended = cv2.addWeighted(live_canvas, alpha, reference, 1.0 - alpha, 0)
    output = reference.copy()
    output[live_mask > 0] = blended[live_mask > 0]
    return output


def draw_crosshair(image, color=(0, 255, 255)):
    """Draw a center crosshair for rough manual alignment."""
    output = image.copy()
    height, width = output.shape[:2]
    center_x = width // 2
    center_y = height // 2
    length = min(width, height) // 12

    cv2.line(output, (center_x - length, center_y), (center_x + length, center_y), color, 2)
    cv2.line(output, (center_x, center_y - length), (center_x, center_y + length), color, 2)
    cv2.circle(output, (center_x, center_y), 5, color, -1)

    return output


def create_rect_mask(image_shape, roi):
    """Create a binary mask from an ROI tuple: x, y, width, height."""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    if roi is None:
        mask[:, :] = 255
        return mask

    x, y, width, height = roi
    if width <= 0 or height <= 0:
        mask[:, :] = 255
        return mask

    mask[y:y + height, x:x + width] = 255
    return mask


def select_reference_roi(reference):
    """
    Let the user select a target skin ROI on the reference image.

    This is one of the most important safeguards against matching the table,
    cables, clothing, or room background instead of the target skin site.
    """
    print("Select the skin/body-site ROI on the reference image.")
    print("Press ENTER or SPACE to accept. Press C to cancel ROI selection.")
    roi = cv2.selectROI("Select Reference Skin ROI", reference, showCrosshair=True)
    cv2.destroyWindow("Select Reference Skin ROI")

    x, y, width, height = [int(value) for value in roi]
    if width == 0 or height == 0:
        print("No ROI selected. Matching will use the full reference image.")
        return None

    print(f"Selected ROI: x={x}, y={y}, width={width}, height={height}")
    return x, y, width, height


def put_instruction_text(image, lines):
    """Draw readable helper text on an OpenCV preview image."""
    y = 24
    for line in lines:
        cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24


def select_reference_draw_mask(reference):
    """
    Let the user draw a freehand or polygon mask around the reference target.

    The accepted mask becomes a manual feature-matching region. This is useful
    when automatic skin masking includes the table, room, or other background.
    """
    state = {
        "mask": np.zeros(reference.shape[:2], dtype=np.uint8),
        "drawing": False,
        "erasing": False,
        "brush_radius": DEFAULT_BRUSH_RADIUS,
        "mode": "brush",
        "polygon_points": [],
    }

    def draw_brush(x, y, value):
        cv2.circle(state["mask"], (x, y), state["brush_radius"], value, -1)

    def mouse_callback(event, x, y, _flags, _param):
        if state["mode"] == "polygon":
            if event == cv2.EVENT_LBUTTONDOWN:
                state["polygon_points"].append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN and state["polygon_points"]:
                state["polygon_points"].pop()
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True
            draw_brush(x, y, 255)
        elif event == cv2.EVENT_RBUTTONDOWN:
            state["erasing"] = True
            draw_brush(x, y, 0)
        elif event == cv2.EVENT_MOUSEMOVE:
            if state["drawing"]:
                draw_brush(x, y, 255)
            elif state["erasing"]:
                draw_brush(x, y, 0)
        elif event == cv2.EVENT_LBUTTONUP:
            state["drawing"] = False
        elif event == cv2.EVENT_RBUTTONUP:
            state["erasing"] = False

    print("Draw a reference mask around the hand/skin target.")
    print("Brush mode: left-drag paints, right-drag erases, [ and ] change brush size.")
    print("Polygon mode: press P, left-click points, U/right-click undo, F fills polygon.")
    print("Press ENTER or SPACE to accept.  Press Q or ESC to cancel.")

    cv2.namedWindow(DRAW_MASK_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(DRAW_MASK_WINDOW, reference.shape[1], min(reference.shape[0], 900))
    cv2.setMouseCallback(DRAW_MASK_WINDOW, mouse_callback)

    accepted = False
    cancelled = False

    while True:
        preview = draw_mask_overlay(reference, state["mask"], color=(0, 255, 255), alpha=0.35)

        if state["mode"] == "polygon" and state["polygon_points"]:
            points = state["polygon_points"]
            for point in points:
                cv2.circle(preview, point, 4, (0, 0, 255), -1)
            for start, end in zip(points, points[1:]):
                cv2.line(preview, start, end, (0, 0, 255), 2)
            if len(points) > 2:
                cv2.line(preview, points[-1], points[0], (0, 0, 255), 1)

        mask_pixels = int(cv2.countNonZero(state["mask"]))
        put_instruction_text(
            preview,
            [
                f"Mode: {state['mode']} | Mask pixels: {mask_pixels}",
                "Brush: left paint, right erase, [/] brush size",
                "Polygon: P toggle, left add, U/right undo, F fill",
                "ENTER/SPACE accept | C clear | Q/ESC cancel",
            ],
        )

        if state["mode"] == "brush":
            cv2.circle(
                preview,
                (reference.shape[1] - 40, 40),
                state["brush_radius"],
                (0, 255, 255),
                2,
            )

        cv2.imshow(DRAW_MASK_WINDOW, preview)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 32):
            if state["mode"] == "polygon" and len(state["polygon_points"]) >= 3:
                polygon = np.array(state["polygon_points"], dtype=np.int32)
                cv2.fillPoly(state["mask"], [polygon], 255)
                state["polygon_points"] = []
            accepted = True
            break
        if key in (27, ord("q")):
            cancelled = True
            break
        if key == ord("c"):
            state["mask"][:, :] = 0
            state["polygon_points"] = []
        if key == ord("p"):
            state["mode"] = "polygon" if state["mode"] == "brush" else "brush"
            state["drawing"] = False
            state["erasing"] = False
        if key == ord("f") and state["mode"] == "polygon" and len(state["polygon_points"]) >= 3:
            polygon = np.array(state["polygon_points"], dtype=np.int32)
            cv2.fillPoly(state["mask"], [polygon], 255)
            state["polygon_points"] = []
        if key == ord("u") and state["polygon_points"]:
            state["polygon_points"].pop()
        if key == ord("["):
            state["brush_radius"] = max(MIN_BRUSH_RADIUS, state["brush_radius"] - 3)
        if key == ord("]"):
            state["brush_radius"] = min(MAX_BRUSH_RADIUS, state["brush_radius"] + 3)

    cv2.destroyWindow(DRAW_MASK_WINDOW)

    if cancelled:
        print("Drawn mask cancelled. Matching will not use a drawn reference mask.")
        return None

    if not accepted or cv2.countNonZero(state["mask"]) == 0:
        print("No drawn mask selected. Matching will not use a drawn reference mask.")
        return None

    print(f"Drawn reference mask accepted: {cv2.countNonZero(state['mask'])} pixels.")
    return state["mask"]


def create_skin_color_mask(image):
    """
    Create a simple skin-color mask.

    This is not a medical or diagnostic segmentation step. It is only a rough
    way to reduce background features before ORB matching.
    """
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Common broad skin-color ranges. These are intentionally loose because
    # lighting and camera white balance can shift skin appearance.
    ycrcb_mask = cv2.inRange(
        ycrcb,
        np.array([0, 133, 77], dtype=np.uint8),
        np.array([255, 180, 135], dtype=np.uint8),
    )
    hsv_mask = cv2.inRange(
        hsv,
        np.array([0, 15, 40], dtype=np.uint8),
        np.array([35, 180, 255], dtype=np.uint8),
    )

    mask = cv2.bitwise_or(ycrcb_mask, hsv_mask)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    return mask


def keep_likely_target_component(mask):
    """
    Keep the most likely hand/skin target component from a rough skin mask.

    Simple color masks often include wood, paper, or other warm-colored
    background. This step closes nearby skin regions and keeps one large,
    central connected component so background patches are less likely to drive
    feature matching.
    """
    if mask is None:
        return None

    height, width = mask.shape[:2]
    kernel_size = max(9, min(width, height) // 35)
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned)
    if count <= 1:
        return cleaned

    image_area = height * width
    image_center = np.array([width / 2.0, height / 2.0])
    max_distance = np.linalg.norm(image_center)

    best_label = None
    best_score = 0.0

    for label in range(1, count):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < image_area * 0.01:
            continue

        centroid = centroids[label]
        distance = np.linalg.norm(centroid - image_center)
        center_weight = 1.0 - min(distance / max_distance, 1.0)
        score = area * (0.5 + center_weight)

        if score > best_score:
            best_score = score
            best_label = label

    if best_label is None:
        return cleaned

    target = np.zeros_like(mask)
    target[labels == best_label] = 255
    target = cv2.morphologyEx(target, cv2.MORPH_CLOSE, kernel, iterations=1)
    return target


def create_auto_target_mask(image):
    """Create an automatic best-guess mask for the main skin/body region."""
    rough_skin_mask = create_skin_color_mask(image)
    return keep_likely_target_component(rough_skin_mask)


def combine_masks(*masks):
    """Combine binary masks by intersection, ignoring None masks."""
    valid_masks = [mask for mask in masks if mask is not None]

    if not valid_masks:
        return None

    combined = valid_masks[0].copy()
    for mask in valid_masks[1:]:
        combined = cv2.bitwise_and(combined, mask)

    return combined


def draw_mask_overlay(image, mask, color=(0, 255, 255), alpha=0.25):
    """Lightly tint the active matching area for debugging."""
    if mask is None:
        return image.copy()

    output = image.copy()
    color_layer = np.zeros_like(output)
    color_layer[:, :] = color
    tinted = cv2.addWeighted(output, 1.0 - alpha, color_layer, alpha, 0)
    output[mask > 0] = tinted[mask > 0]
    return output


def build_feature_mask(image, base_mask=None, skin_enabled=False, auto_target_enabled=True):
    """
    Build the mask that controls where ORB is allowed to find features.

    base_mask can be a manually selected ROI or the live-video placement mask.
    auto_target_enabled keeps the largest likely skin/body component.
    skin_enabled applies the broader skin-color mask.
    """
    masks = []

    if base_mask is not None:
        masks.append(base_mask)

    if auto_target_enabled:
        masks.append(create_auto_target_mask(image))
    elif skin_enabled:
        masks.append(create_skin_color_mask(image))

    if not masks:
        return None

    return combine_masks(*masks)


def build_reference_feature_mask(reference, manual_mask, skin_enabled, auto_target_enabled):
    """Build the reference feature mask, treating a user-drawn mask as authoritative."""
    if manual_mask is not None:
        return manual_mask.copy()

    return build_feature_mask(
        reference,
        base_mask=None,
        skin_enabled=skin_enabled,
        auto_target_enabled=auto_target_enabled,
    )


def shift_image_and_mask(image, mask, dx, dy):
    """Shift an image and mask without wrapping pixels around the canvas."""
    height, width = image.shape[:2]
    shifted_image = np.zeros_like(image)
    shifted_mask = np.zeros_like(mask)

    src_x1 = max(0, -dx)
    src_y1 = max(0, -dy)
    src_x2 = min(width, width - dx)
    src_y2 = min(height, height - dy)

    dst_x1 = max(0, dx)
    dst_y1 = max(0, dy)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    if src_x2 > src_x1 and src_y2 > src_y1:
        shifted_image[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]
        shifted_mask[dst_y1:dst_y2, dst_x1:dst_x2] = mask[src_y1:src_y2, src_x1:src_x2]

    return shifted_image, shifted_mask


def visual_overlap_score(reference, live_canvas, live_mask, reference_mask=None, score_scale=DEFAULT_VISUAL_SCORE_SCALE):
    """
    Score direct reference/live visual overlap inside the active ROI.

    This is a feedback score for overlay mode, not a diagnostic decision. It
    blends edge similarity with normalized grayscale similarity so lighting
    changes do not dominate as strongly.
    """
    active_mask = live_mask.copy()
    if reference_mask is not None:
        active_mask = cv2.bitwise_and(active_mask, reference_mask)

    if cv2.countNonZero(active_mask) < 200:
        return None

    if 0 < score_scale < 1.0:
        size = (
            max(1, int(reference.shape[1] * score_scale)),
            max(1, int(reference.shape[0] * score_scale)),
        )
        ref_small = cv2.resize(reference, size, interpolation=cv2.INTER_AREA)
        live_small = cv2.resize(live_canvas, size, interpolation=cv2.INTER_AREA)
        mask_small = cv2.resize(active_mask, size, interpolation=cv2.INTER_NEAREST)
    else:
        ref_small = reference
        live_small = live_canvas
        mask_small = active_mask

    if cv2.countNonZero(mask_small) < 50:
        return None

    ref_gray = cv2.cvtColor(ref_small, cv2.COLOR_BGR2GRAY)
    live_gray = cv2.cvtColor(live_small, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ref_norm = clahe.apply(ref_gray)
    live_norm = clahe.apply(live_gray)

    ref_edges = cv2.Canny(ref_norm, 40, 120)
    live_edges = cv2.Canny(live_norm, 40, 120)

    mask_bool = mask_small > 0
    ref_values = ref_norm[mask_bool].astype(np.float32)
    live_values = live_norm[mask_bool].astype(np.float32)

    ref_values -= float(np.mean(ref_values))
    live_values -= float(np.mean(live_values))
    denom = float(np.linalg.norm(ref_values) * np.linalg.norm(live_values))
    gray_corr = 0.0 if denom <= 1e-6 else float(np.dot(ref_values, live_values) / denom)
    gray_score = np.clip((gray_corr + 1.0) * 50.0, 0.0, 100.0)

    ref_edge_bool = (ref_edges > 0) & mask_bool
    live_edge_bool = (live_edges > 0) & mask_bool
    union = np.count_nonzero(ref_edge_bool | live_edge_bool)
    if union == 0:
        edge_score = 0.0
    else:
        edge_score = 100.0 * np.count_nonzero(ref_edge_bool & live_edge_bool) / union

    return float((0.55 * gray_score) + (0.45 * edge_score))


def compute_visual_overlay_summary(
    reference,
    live_canvas,
    live_mask,
    reference_mask=None,
    score_scale=DEFAULT_VISUAL_SCORE_SCALE,
):
    """Compute current visual overlap plus a small best-direction nudge."""
    current_score = visual_overlap_score(reference, live_canvas, live_mask, reference_mask, score_scale)
    if current_score is None:
        return {
            "score": 0,
            "status": "No ROI overlap",
            "hint": "Position uncertain: use overlay manually",
        }

    candidates = [
        ("Move camera left", -VISUAL_NUDGE_PIXELS, 0),
        ("Move camera right", VISUAL_NUDGE_PIXELS, 0),
        ("Move camera up", 0, -VISUAL_NUDGE_PIXELS),
        ("Move camera down", 0, VISUAL_NUDGE_PIXELS),
    ]

    best_hint = "Overlay position close"
    best_score = current_score

    for hint, dx, dy in candidates:
        shifted_live, shifted_mask = shift_image_and_mask(live_canvas, live_mask, dx, dy)
        candidate_score = visual_overlap_score(
            reference,
            shifted_live,
            shifted_mask,
            reference_mask,
            score_scale,
        )
        if candidate_score is not None and candidate_score > best_score + 2.0:
            best_score = candidate_score
            best_hint = hint

    score = int(round(np.clip(current_score, 0.0, 100.0)))
    if score >= 75:
        status = "Strong overlap"
    elif score >= 55:
        status = "Moderate overlap"
    else:
        status = "Weak overlap"

    return {
        "score": score,
        "status": status,
        "hint": best_hint,
        "best_score": int(round(np.clip(best_score, 0.0, 100.0))),
    }


def draw_status_text(
    image,
    mode,
    alpha,
    scale,
    x_offset,
    y_offset,
    alignment_summary=None,
    visual_summary=None,
):
    """Draw current mode, controls, and optional overlay alignment scores."""
    output = image.copy()
    lines = [
        f"Mode: {mode}",
        f"Overlay opacity: {alpha:.2f}",
        f"Live scale: {scale}%",
        f"Offset: x={x_offset}, y={y_offset}",
    ]

    if visual_summary is not None:
        lines.extend(
            [
                f"Visual overlap: {visual_summary['score']}/100 ({visual_summary['status']})",
                f"Overlay nudge: {visual_summary['hint']}",
            ]
        )

    if alignment_summary is not None:
        lines.extend(
            [
                f"Landmark score: {alignment_summary['score']}/100 ({alignment_summary['status']})",
                f"Landmark guidance: {alignment_summary['hint']}",
                f"Landmark evidence: {alignment_summary['inliers']} inliers / {alignment_summary['matches']} matches",
            ]
        )

        if alignment_summary.get("scale") is not None:
            lines.append(
                f"Scale={alignment_summary['scale']:.2f}, rotation={alignment_summary['rotation']:.1f} deg"
            )

    lines.append("Keys: O overlay/side-by-side | M matches | B ORB/markers | S save | R reset | T text | Q quit")

    y = 28
    for line in lines:
        cv2.putText(output, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(output, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        y += 28

    return output


def preprocess_for_features(image):
    """Convert image to a contrast-normalized grayscale image for ORB."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.GaussianBlur(enhanced, (3, 3), 0)


def create_orb_detector(max_features):
    """Create the ORB detector used for rough reference/live matching."""
    return cv2.ORB_create(
        nfeatures=max_features,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=15,
        fastThreshold=12,
    )


def detect_orb_features(orb, image, mask=None):
    """Detect ORB keypoints/descriptors in an image."""
    processed = preprocess_for_features(image)
    return orb.detectAndCompute(processed, mask)


def detect_marker_blobs(
    image,
    mask=None,
    max_markers=DEFAULT_MAX_MARKERS,
    detect_scale=DEFAULT_MARKER_DETECT_SCALE,
):
    """
    Detect marker-drawn dot landmarks as colored blob centers.

    This is intended for calibration/fake-arm experiments where the landmarks
    are marker dots. It is not a medical lesion classifier.
    """
    if 0 < detect_scale < 1.0:
        working_image = cv2.resize(image, None, fx=detect_scale, fy=detect_scale, interpolation=cv2.INTER_AREA)
        working_mask = None
        if mask is not None:
            working_mask = cv2.resize(mask, (working_image.shape[1], working_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        coordinate_scale = 1.0 / detect_scale
    else:
        working_image = image
        working_mask = mask
        coordinate_scale = 1.0

    hsv = cv2.cvtColor(working_image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bgr = working_image.astype(np.float32)
    b, g, r = cv2.split(bgr)

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    orange_red = (
        (((hue <= 24) | (hue >= 170)) & (saturation >= 55) & (value >= 35))
    )
    red_signal = r - (0.55 * g + 0.45 * b)
    red_brown = (red_signal > 35) & (saturation >= 35)
    candidate_mask = (orange_red | red_brown).astype(np.uint8) * 255

    if working_mask is not None:
        candidate_mask = cv2.bitwise_and(candidate_mask, working_mask)

    candidate_mask = cv2.medianBlur(candidate_mask, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _hierarchy = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = working_image.shape[0] * working_image.shape[1]
    min_area = max(6, int(image_area * 0.000004))
    max_area = max(120, int(image_area * 0.003))
    local_background = cv2.GaussianBlur(gray, (0, 0), 13)
    red_signal_float = red_signal.astype(np.float32)
    markers = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w < 3 or h < 3:
            continue

        aspect = max(w, h) / max(1.0, min(w, h))
        if aspect > 3.0:
            continue

        perimeter = cv2.arcLength(contour, True)
        circularity = (4.0 * math.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0.0
        if circularity < 0.20:
            continue

        component_mask = np.zeros(working_image.shape[:2], dtype=np.uint8)
        cv2.drawContours(component_mask, [contour], -1, 255, -1)
        if working_mask is not None and cv2.countNonZero(cv2.bitwise_and(component_mask, working_mask)) < 0.80 * cv2.countNonZero(component_mask):
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue

        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])

        cx_i = int(np.clip(round(cx), 0, working_image.shape[1] - 1))
        cy_i = int(np.clip(round(cy), 0, working_image.shape[0] - 1))
        dark_contrast = max(0.0, float(local_background[cy_i, cx_i] - gray[cy_i, cx_i]))
        mean_sat = cv2.mean(saturation, mask=component_mask)[0]
        mean_red_signal = cv2.mean(red_signal_float, mask=component_mask)[0]
        score = (0.60 * area) + (1.20 * mean_sat) + (1.50 * max(0.0, mean_red_signal)) + dark_contrast
        box_half = int(np.clip(max(w, h) * 0.9 + 8, 9, 34))

        markers.append(
            {
                "center": (cx, cy),
                "bbox": (x, y, w, h),
                "area": area,
                "score": float(score),
                "box_half": box_half,
            }
        )

    markers.sort(key=lambda marker: marker["score"], reverse=True)
    selected = []
    for marker in markers[:max_markers]:
        x, y, w, h = marker["bbox"]
        cx, cy = marker["center"]
        selected.append(
            {
                "center": (cx * coordinate_scale, cy * coordinate_scale),
                "bbox": (
                    int(round(x * coordinate_scale)),
                    int(round(y * coordinate_scale)),
                    max(1, int(round(w * coordinate_scale))),
                    max(1, int(round(h * coordinate_scale))),
                ),
                "area": float(marker["area"] * coordinate_scale * coordinate_scale),
                "score": marker["score"],
                "box_half": int(round(marker["box_half"] * coordinate_scale)),
            }
        )

    if coordinate_scale != 1.0:
        candidate_mask = cv2.resize(candidate_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    return selected, candidate_mask


def markers_to_keypoints(markers):
    """Convert marker center dictionaries into OpenCV keypoints."""
    return [
        cv2.KeyPoint(float(marker["center"][0]), float(marker["center"][1]), float(marker["box_half"] * 2))
        for marker in markers
    ]


def estimate_similarity_from_point_pair(ref_a, ref_b, live_a, live_b):
    """Estimate a 2D similarity transform from two point correspondences."""
    ref_a = np.array(ref_a, dtype=np.float32)
    ref_b = np.array(ref_b, dtype=np.float32)
    live_a = np.array(live_a, dtype=np.float32)
    live_b = np.array(live_b, dtype=np.float32)

    ref_vec = ref_b - ref_a
    live_vec = live_b - live_a
    ref_len = float(np.linalg.norm(ref_vec))
    live_len = float(np.linalg.norm(live_vec))

    if ref_len < 20.0 or live_len < 20.0:
        return None

    scale = live_len / ref_len
    if scale < MIN_VALID_SCALE or scale > MAX_VALID_SCALE:
        return None

    ref_angle = math.atan2(float(ref_vec[1]), float(ref_vec[0]))
    live_angle = math.atan2(float(live_vec[1]), float(live_vec[0]))
    theta = live_angle - ref_angle
    cos_t = math.cos(theta) * scale
    sin_t = math.sin(theta) * scale

    matrix = np.array(
        [
            [cos_t, -sin_t, 0.0],
            [sin_t, cos_t, 0.0],
        ],
        dtype=np.float32,
    )
    matrix[:, 2] = live_a - matrix[:, :2] @ ref_a
    return matrix


def project_points(points, matrix):
    """Apply a 2x3 affine matrix to a list of points."""
    pts = np.array(points, dtype=np.float32)
    ones = np.ones((len(pts), 1), dtype=np.float32)
    homogeneous = np.hstack([pts, ones])
    return homogeneous @ matrix.T


def match_marker_constellation(reference_markers, live_markers, max_projected_distance=42.0):
    """
    Match marker-dot centers by geometric constellation instead of descriptor appearance.

    The dots can look almost identical, so this tries many two-dot transforms and
    keeps the transform that explains the most one-to-one dot correspondences.
    """
    if len(reference_markers) < 2 or len(live_markers) < 2:
        return [], None, []

    reference_points = np.float32([marker["center"] for marker in reference_markers])
    live_points = np.float32([marker["center"] for marker in live_markers])
    best_pairs = []
    best_matrix = None
    best_error = float("inf")
    max_ref_hypotheses = min(MAX_MARKER_HYPOTHESIS_POINTS, len(reference_points))
    max_live_hypotheses = min(MAX_MARKER_HYPOTHESIS_POINTS, len(live_points))
    reference_hypothesis_indices = range(max_ref_hypotheses)
    live_hypothesis_indices = range(max_live_hypotheses)

    for i in reference_hypothesis_indices:
        for j in reference_hypothesis_indices:
            if i >= j:
                continue
            for a in live_hypothesis_indices:
                for b in live_hypothesis_indices:
                    if a == b:
                        continue

                    matrix = estimate_similarity_from_point_pair(
                        reference_points[i],
                        reference_points[j],
                        live_points[a],
                        live_points[b],
                    )
                    if matrix is None:
                        continue

                    projected = project_points(reference_points, matrix)
                    distances = np.linalg.norm(
                        projected[:, None, :] - live_points[None, :, :],
                        axis=2,
                    )
                    ref_indices, live_indices = np.where(distances <= max_projected_distance)
                    candidates = [
                        (float(distances[ref_index, live_index]), int(ref_index), int(live_index))
                        for ref_index, live_index in zip(ref_indices, live_indices)
                    ]

                    used_ref = set()
                    used_live = set()
                    pairs = []
                    for distance, ref_index, live_index in sorted(candidates, key=lambda item: item[0]):
                        if ref_index in used_ref or live_index in used_live:
                            continue
                        used_ref.add(ref_index)
                        used_live.add(live_index)
                        pairs.append((ref_index, live_index, distance))

                    if len(pairs) < 2:
                        continue

                    avg_error = float(np.mean([pair[2] for pair in pairs]))
                    score = (-100.0 * len(pairs)) + avg_error
                    best_score = (-100.0 * len(best_pairs)) + best_error
                    if score < best_score:
                        best_pairs = pairs
                        best_matrix = matrix
                        best_error = avg_error

    if len(best_pairs) >= 3:
        ref_fit = np.float32([reference_points[ref_idx] for ref_idx, _live_idx, _distance in best_pairs])
        live_fit = np.float32([live_points[live_idx] for _ref_idx, live_idx, _distance in best_pairs])
        refined, inlier_mask = cv2.estimateAffinePartial2D(
            ref_fit,
            live_fit,
            method=cv2.RANSAC,
            ransacReprojThreshold=max_projected_distance,
            maxIters=1000,
            confidence=0.98,
        )
        if refined is not None and inlier_mask is not None:
            inlier_pairs = [
                pair for pair, keep in zip(best_pairs, inlier_mask.ravel())
                if keep
            ]
            if len(inlier_pairs) >= 2:
                best_pairs = inlier_pairs
                best_matrix = refined.astype(np.float32)

    pseudo_matches = [
        cv2.DMatch(_queryIdx=int(ref_index), _trainIdx=int(live_index), _distance=float(distance))
        for ref_index, live_index, distance in best_pairs
    ]
    return pseudo_matches, best_matrix, best_pairs


def match_orb_descriptors(reference_descriptors, live_descriptors):
    """Return sorted ORB descriptor matches using a ratio test."""
    if reference_descriptors is None or live_descriptors is None:
        return []

    if len(reference_descriptors) == 0 or len(live_descriptors) == 0:
        return []

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    match_pairs = matcher.knnMatch(reference_descriptors, live_descriptors, k=2)

    good_matches = []
    for pair in match_pairs:
        if len(pair) < 2:
            continue
        best, second_best = pair
        if best.distance < 0.78 * second_best.distance:
            good_matches.append(best)

    return sorted(good_matches, key=lambda match: match.distance)


def filter_matches_with_affine_ransac(reference_keypoints, live_keypoints, matches):
    """
    Keep only matches that agree with one rough affine transform.

    This rejects many accidental background/texture matches before computing
    movement guidance.
    """
    if len(matches) < 4:
        return matches, None

    reference_points = np.float32([
        reference_keypoints[match.queryIdx].pt for match in matches
    ])
    live_points = np.float32([
        live_keypoints[match.trainIdx].pt for match in matches
    ])

    transform, inlier_mask = cv2.estimateAffinePartial2D(
        reference_points,
        live_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=8.0,
        maxIters=2000,
        confidence=0.98,
    )

    if inlier_mask is None:
        return matches, transform

    inlier_matches = [
        match for match, is_inlier in zip(matches, inlier_mask.ravel())
        if is_inlier
    ]

    return inlier_matches, transform


def estimate_translation_from_matches(reference_keypoints, live_keypoints, matches):
    """
    Estimate rough x/y shift from matched feature points.

    Positive dx means the live feature points are to the right of the reference
    feature points in the current display canvas. Positive dy means lower.
    """
    if len(matches) < 4:
        return None

    dx_values = []
    dy_values = []

    for match in matches:
        ref_x, ref_y = reference_keypoints[match.queryIdx].pt
        live_x, live_y = live_keypoints[match.trainIdx].pt
        dx_values.append(live_x - ref_x)
        dy_values.append(live_y - ref_y)

    return float(np.median(dx_values)), float(np.median(dy_values))


def movement_hint_from_translation(translation, tolerance=25):
    """Convert rough x/y shift into early movement guidance text."""
    if translation is None:
        return "Not enough matches"

    dx, dy = translation
    hints = []

    if dx > tolerance:
        hints.append("Move camera right")
    elif dx < -tolerance:
        hints.append("Move camera left")

    if dy > tolerance:
        hints.append("Move camera down")
    elif dy < -tolerance:
        hints.append("Move camera up")

    if not hints:
        return "Rough x/y alignment good"

    return " + ".join(hints)


def analyze_affine_transform(transform):
    """
    Extract translation, rotation, and scale from an affine transform.

    The transform maps reference-image points to live-image points.
    """
    if transform is None:
        return None

    a = transform[0, 0]
    c = transform[1, 0]
    tx = transform[0, 2]
    ty = transform[1, 2]

    scale = float(np.sqrt(a * a + c * c))
    rotation_degrees = float(np.degrees(np.arctan2(c, a)))

    return {
        "dx": float(tx),
        "dy": float(ty),
        "scale": scale,
        "rotation_degrees": rotation_degrees,
    }


def movement_hint_from_alignment(alignment, inlier_count, total_match_count=0):
    """Convert affine alignment values into user-facing guidance."""
    if alignment is None:
        return "Target not found"

    if inlier_count < MIN_INLIERS_FOR_GUIDANCE:
        return "Target not found / not enough reliable matches"

    inlier_ratio = inlier_count / max(total_match_count, 1)

    if inlier_ratio < MIN_INLIER_RATIO:
        return "Target not found / weak match quality"

    dx = alignment["dx"]
    dy = alignment["dy"]
    scale = alignment["scale"]
    rotation = alignment["rotation_degrees"]

    translation = np.sqrt(dx * dx + dy * dy)

    if scale < MIN_VALID_SCALE or scale > MAX_VALID_SCALE:
        return "Invalid match / target likely not in frame"

    if abs(rotation) > MAX_VALID_ROTATION:
        return "Invalid match / rotation too large"

    if translation > MAX_VALID_TRANSLATION:
        return "Invalid match / target likely not in frame"

    hints = []

    if dx > 25:
        hints.append("Move camera right")
    elif dx < -25:
        hints.append("Move camera left")

    if dy > 25:
        hints.append("Move camera down")
    elif dy < -25:
        hints.append("Move camera up")

    if scale > 1.08:
        hints.append("Move farther")
    elif scale < 0.92:
        hints.append("Move closer")

    if rotation > 5:
        hints.append("Rotate clockwise")
    elif rotation < -5:
        hints.append("Rotate counterclockwise")

    if not hints:
        return "Aligned / close alignment"

    return " + ".join(hints)

def score_alignment(alignment, inlier_count, total_match_count):
    """Create a rough 0-100 alignment score from transform and match quality."""
    if alignment is None:
        return 0, "Target not found"

    if inlier_count < MIN_INLIERS_FOR_GUIDANCE:
        return 0, "Target not found"

    inlier_ratio = inlier_count / max(total_match_count, 1)

    if inlier_ratio < MIN_INLIER_RATIO:
        return 0, "Weak match"

    scale = alignment["scale"]
    rotation = abs(alignment["rotation_degrees"])
    translation_error = np.sqrt(alignment["dx"] ** 2 + alignment["dy"] ** 2)

    if scale < MIN_VALID_SCALE or scale > MAX_VALID_SCALE:
        return 0, "Invalid scale"

    if rotation > MAX_VALID_ROTATION:
        return 0, "Invalid rotation"

    if translation_error > MAX_VALID_TRANSLATION:
        return 0, "Target likely not in frame"

    scale_error = abs(scale - 1.0)

    translation_penalty = min(40, translation_error / 3.0)
    rotation_penalty = min(25, rotation * 2.0)
    scale_penalty = min(25, scale_error * 200.0)
    inlier_penalty = max(0, 15 - inlier_count) * 1.5
    ratio_penalty = max(0, 0.45 - inlier_ratio) * 30.0

    score = 100 - (
        translation_penalty
        + rotation_penalty
        + scale_penalty
        + inlier_penalty
        + ratio_penalty
    )


    score = int(max(0, min(100, score)))

    if score >= 80:
        status = "Aligned"
    elif score >= 55:
        status = "Close"
    else:
        status = "Not aligned"

    return score, status


def alignment_is_reliable(alignment, inlier_count, total_match_count):
    """Return True only when the affine transform is strong enough to display."""
    if alignment is None:
        return False

    if inlier_count < MIN_INLIERS_FOR_GUIDANCE:
        return False

    inlier_ratio = inlier_count / max(total_match_count, 1)
    if inlier_ratio < MIN_INLIER_RATIO:
        return False

    dx = alignment["dx"]
    dy = alignment["dy"]
    scale = alignment["scale"]
    rotation = alignment["rotation_degrees"]
    translation = np.sqrt(dx * dx + dy * dy)

    if scale < MIN_VALID_SCALE or scale > MAX_VALID_SCALE:
        return False

    if abs(rotation) > MAX_VALID_ROTATION:
        return False

    if translation > MAX_VALID_TRANSLATION:
        return False

    return True


def marker_alignment_is_reliable(alignment, inlier_count, total_marker_count):
    """Return True when marker-dot geometry is strong enough for guidance."""
    if alignment is None:
        return False

    if inlier_count < MIN_MARKER_INLIERS_FOR_GUIDANCE:
        return False

    inlier_ratio = inlier_count / max(total_marker_count, 1)
    if inlier_ratio < MIN_MARKER_INLIER_RATIO:
        return False

    scale = alignment["scale"]
    rotation = alignment["rotation_degrees"]
    dx = alignment["dx"]
    dy = alignment["dy"]
    translation = np.sqrt(dx * dx + dy * dy)

    if scale < MIN_VALID_SCALE or scale > MAX_VALID_SCALE:
        return False

    if abs(rotation) > MAX_VALID_ROTATION:
        return False

    if translation > MAX_VALID_TRANSLATION:
        return False

    return True


def marker_score_alignment(alignment, inlier_count, total_marker_count):
    """Score marker-dot alignment with thresholds suited to marker centers."""
    if not marker_alignment_is_reliable(alignment, inlier_count, total_marker_count):
        return 0, "Target not found"

    translation_error = np.sqrt(alignment["dx"] ** 2 + alignment["dy"] ** 2)
    rotation_error = abs(alignment["rotation_degrees"])
    scale_error = abs(alignment["scale"] - 1.0)
    inlier_ratio = inlier_count / max(total_marker_count, 1)

    score = 100 - (
        min(45, translation_error / 3.0)
        + min(20, rotation_error * 2.0)
        + min(20, scale_error * 180.0)
        + max(0, 0.75 - inlier_ratio) * 25.0
    )
    score = int(max(0, min(100, score)))

    if score >= 80:
        status = "Aligned"
    elif score >= 55:
        status = "Close"
    else:
        status = "Not aligned"

    return score, status


def compute_orb_alignment_summary(
    live_canvas,
    live_feature_mask,
    orb,
    reference_keypoints,
    reference_descriptors,
):
    """Compute a compact alignment score for overlay mode using ORB matches."""
    live_keypoints, live_descriptors = detect_orb_features(orb, live_canvas, live_feature_mask)
    matches = match_orb_descriptors(reference_descriptors, live_descriptors)
    inlier_matches, transform = filter_matches_with_affine_ransac(
        reference_keypoints,
        live_keypoints,
        matches,
    )
    alignment = analyze_affine_transform(transform)
    reliable = alignment_is_reliable(alignment, len(inlier_matches), len(matches))
    score, status = score_alignment(alignment, len(inlier_matches), len(matches))
    hint = movement_hint_from_alignment(alignment, len(inlier_matches), len(matches))

    if not reliable:
        score = 0
        status = "Target not found"

    return {
        "score": score,
        "status": status,
        "hint": hint,
        "matches": len(matches),
        "inliers": len(inlier_matches),
        "scale": None if not reliable else alignment["scale"],
        "rotation": None if not reliable else alignment["rotation_degrees"],
    }


def compute_marker_alignment_summary(
    live_canvas,
    live_mask,
    reference_markers,
    max_markers,
    marker_detect_scale,
):
    """Compute a compact alignment score for overlay mode using marker blobs."""
    live_markers, _live_marker_mask = detect_marker_blobs(
        live_canvas,
        live_mask,
        max_markers,
        detect_scale=marker_detect_scale,
    )
    matches, transform, _pairs = match_marker_constellation(reference_markers, live_markers)
    alignment = analyze_affine_transform(transform)
    total_marker_count = min(len(reference_markers), len(live_markers))
    reliable = marker_alignment_is_reliable(alignment, len(matches), total_marker_count)
    score, status = marker_score_alignment(alignment, len(matches), total_marker_count)

    if alignment is None:
        hint = "Target not found / not enough marker geometry"
    elif reliable:
        hint = movement_hint_from_alignment(
            alignment,
            MIN_INLIERS_FOR_GUIDANCE,
            MIN_INLIERS_FOR_GUIDANCE,
        )
    else:
        hint = "Target not found / not enough reliable marker matches"
        score = 0
        status = "Target not found"

    return {
        "score": score,
        "status": status,
        "hint": hint,
        "matches": len(matches),
        "inliers": len(matches),
        "reference_markers": len(reference_markers),
        "live_markers": len(live_markers),
        "scale": None if not reliable else alignment["scale"],
        "rotation": None if not reliable else alignment["rotation_degrees"],
    }


def draw_marker_debug_view(
    reference,
    live_canvas,
    reference_mask,
    live_feature_mask,
    reference_markers,
    max_matches_to_draw,
    mask_label,
    max_markers,
    marker_detect_scale,
):
    """Build a side-by-side marker-dot match visualization and guidance text."""
    live_markers, live_marker_mask = detect_marker_blobs(
        live_canvas,
        live_feature_mask,
        max_markers,
        detect_scale=marker_detect_scale,
    )
    reference_keypoints = markers_to_keypoints(reference_markers)
    live_keypoints = markers_to_keypoints(live_markers)
    matches, transform, _pairs = match_marker_constellation(reference_markers, live_markers)
    drawn_matches = matches[:max_matches_to_draw]
    alignment = analyze_affine_transform(transform)
    total_marker_count = min(len(reference_markers), len(live_markers))
    reliable_alignment = marker_alignment_is_reliable(alignment, len(matches), total_marker_count)

    if alignment is None:
        hint = "Target not found / not enough marker geometry"
        score, status = 0, "Target not found"
        shift_text = "Rough shift: unavailable"
        transform_text = "Scale/rotation: unavailable"
    else:
        hint = movement_hint_from_alignment(alignment, MIN_INLIERS_FOR_GUIDANCE, MIN_INLIERS_FOR_GUIDANCE)
        score, status = marker_score_alignment(alignment, len(matches), total_marker_count)
        if reliable_alignment:
            shift_text = f"Rough shift: dx={alignment['dx']:.1f}, dy={alignment['dy']:.1f}"
            transform_text = (
                f"Scale={alignment['scale']:.2f}, "
                f"rotation={alignment['rotation_degrees']:.1f} deg"
            )
        else:
            hint = "Target not found / not enough reliable marker matches"
            shift_text = "Rough shift: unavailable"
            transform_text = "Scale/rotation: unreliable until target is found"

    reference_debug = draw_mask_overlay(reference, reference_mask, color=(0, 255, 255))
    live_debug = draw_mask_overlay(live_canvas, live_feature_mask, color=(0, 255, 255))

    for marker in reference_markers:
        x, y, w, h = marker["bbox"]
        cv2.rectangle(reference_debug, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cx, cy = marker["center"]
        cv2.circle(reference_debug, (int(round(cx)), int(round(cy))), 3, (0, 0, 255), -1)

    for marker in live_markers:
        x, y, w, h = marker["bbox"]
        cv2.rectangle(live_debug, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cx, cy = marker["center"]
        cv2.circle(live_debug, (int(round(cx)), int(round(cy))), 3, (0, 0, 255), -1)

    debug_view = cv2.drawMatches(
        reference_debug,
        reference_keypoints,
        live_debug,
        live_keypoints,
        drawn_matches,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(0, 0, 255),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    if drawn_matches:
        average_error = sum(match.distance for match in drawn_matches) / len(drawn_matches)
    else:
        average_error = 0.0

    inlier_ratio = len(matches) / max(total_marker_count, 1)
    lines = [
        "Mode: marker blob matching debug",
        f"Markers: reference={len(reference_markers)} / live={len(live_markers)}",
        f"Matches drawn: {len(drawn_matches)} / inliers: {len(matches)}",
        f"Inlier ratio: {inlier_ratio:.2f} (need >= {MIN_MARKER_INLIER_RATIO:.2f}; inliers need >= {MIN_MARKER_INLIERS_FOR_GUIDANCE})",
        f"Average projected marker error: {average_error:.1f} px",
        f"Matching area: {mask_label}",
        f"Marker detection scale: {marker_detect_scale:.2f}",
        shift_text,
        transform_text,
        f"Alignment score: {score}/100 ({status})",
        f"Guidance: {hint}",
        "Keys: M return | B ORB/markers | A auto target | K skin mask | O overlay/side-by-side | S save | R reset | Q quit",
    ]

    y = 28
    for line in lines:
        cv2.putText(debug_view, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(debug_view, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        y += 28

    return debug_view


def draw_match_debug_view(
    reference,
    live_canvas,
    reference_mask,
    live_feature_mask,
    orb,
    reference_keypoints,
    reference_descriptors,
    max_matches_to_draw,
    mask_label,
):
    """Build a side-by-side ORB match visualization and rough guidance text."""
    live_keypoints, live_descriptors = detect_orb_features(orb, live_canvas, live_feature_mask)
    matches = match_orb_descriptors(reference_descriptors, live_descriptors)
    inlier_matches, transform = filter_matches_with_affine_ransac(
        reference_keypoints,
        live_keypoints,
        matches,
    )
    drawn_matches = inlier_matches[:max_matches_to_draw]
    alignment = analyze_affine_transform(transform)
    reliable_alignment = alignment_is_reliable(alignment, len(inlier_matches), len(matches))

    if alignment is None:
        translation = estimate_translation_from_matches(
            reference_keypoints,
            live_keypoints,
            drawn_matches,
        )
        hint = movement_hint_from_translation(translation)
        score, status = 0, "Not enough matches"
    else:
        hint = movement_hint_from_alignment(alignment, len(inlier_matches), len(matches))
        score, status = score_alignment(alignment, len(inlier_matches), len(matches))
        if reliable_alignment:
            translation = (alignment["dx"], alignment["dy"])
        else:
            translation = None

    reference_debug = draw_mask_overlay(reference, reference_mask, color=(0, 255, 255))
    live_debug = draw_mask_overlay(live_canvas, live_feature_mask, color=(0, 255, 255))

    debug_view = cv2.drawMatches(
        reference_debug,
        reference_keypoints,
        live_debug,
        live_keypoints,
        drawn_matches,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(0, 0, 255),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    if drawn_matches:
        average_distance = sum(match.distance for match in drawn_matches) / len(drawn_matches)
    else:
        average_distance = 0.0

    inlier_ratio = len(inlier_matches) / max(len(matches), 1)

    if translation is None:
        shift_text = "Rough shift: unavailable"
    else:
        dx, dy = translation
        shift_text = f"Rough shift: dx={dx:.1f}, dy={dy:.1f}"

    if alignment is None:
        transform_text = "Scale/rotation: unavailable"
    elif not reliable_alignment:
        transform_text = "Scale/rotation: unreliable until target is found"
    else:
        transform_text = (
            f"Scale={alignment['scale']:.2f}, "
            f"rotation={alignment['rotation_degrees']:.1f} deg"
        )

    lines = [
        "Mode: ORB feature matching debug",
        f"Matches drawn: {len(drawn_matches)} / inliers: {len(inlier_matches)} / total: {len(matches)}",
        f"Inlier ratio: {inlier_ratio:.2f} (need >= {MIN_INLIER_RATIO:.2f}; inliers need >= {MIN_INLIERS_FOR_GUIDANCE})",
        f"Average match distance: {average_distance:.1f} (lower is better)",
        f"Matching area: {mask_label}",
        shift_text,
        transform_text,
        f"Alignment score: {score}/100 ({status})",
        f"Guidance: {hint}",
        "Keys: M return | A auto target | K skin mask | O overlay/side-by-side | S save | R reset | Q quit",
    ]

    y = 28
    for line in lines:
        cv2.putText(debug_view, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(debug_view, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        y += 28

    return debug_view


def resolve_output_dir(output_dir):
    """Return a writable output directory from a GUI or CLI path string."""
    value = str(output_dir or "").strip()
    return Path(value).expanduser() if value else Path.home() / "Desktop" / "output_images"


def save_alignment_outputs(display, live_frame, output_dir):
    """Save the current displayed alignment view and raw live frame."""
    output_path = resolve_output_dir(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    display_path = output_path / f"alignment_view_{timestamp}.jpg"
    live_path = output_path / f"aligned_live_frame_{timestamp}.jpg"

    cv2.imwrite(str(display_path), display)
    cv2.imwrite(str(live_path), live_frame)

    print(f"Saved alignment view: {display_path}")
    print(f"Saved live frame: {live_path}")


WEB_APP_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Live Reference Alignment</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f5f7;
      --panel: #ffffff;
      --ink: #18202a;
      --muted: #667085;
      --line: #d8dde5;
      --accent: #0f766e;
      --accent-dark: #115e59;
      --danger: #b42318;
      --ok: #067647;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      height: 52px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 18px;
      background: #17212b;
      color: #fff;
      border-bottom: 1px solid #101820;
    }
    header h1 {
      margin: 0;
      font-size: 17px;
      font-weight: 650;
      letter-spacing: 0;
    }
    header .status { color: #cbd5e1; font-size: 13px; }
    main {
      display: grid;
      grid-template-columns: minmax(620px, 1fr) 390px;
      gap: 14px;
      padding: 14px;
      height: calc(100vh - 52px);
    }
    .viewer {
      min-height: 0;
      background: #111820;
      border: 1px solid #101820;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }
    #stream {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      display: block;
    }
    .sidebar {
      min-width: 0;
      overflow: auto;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }
    section h2 {
      margin: 0 0 10px;
      font-size: 13px;
      text-transform: uppercase;
      color: #344054;
      letter-spacing: .04em;
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      align-items: center;
    }
    .wide { grid-column: 1 / -1; }
    label {
      display: grid;
      gap: 4px;
      color: var(--muted);
      font-size: 12px;
    }
    input, select, button {
      font: inherit;
      border-radius: 6px;
      border: 1px solid var(--line);
      min-height: 34px;
    }
    input, select {
      width: 100%;
      padding: 6px 8px;
      color: var(--ink);
      background: #fff;
    }
    input[type="range"] { padding: 0; }
    input[type="checkbox"] { min-height: auto; width: auto; }
    .check {
      display: flex;
      gap: 8px;
      align-items: center;
      color: var(--ink);
      font-size: 13px;
    }
    button {
      padding: 6px 10px;
      background: #fff;
      color: var(--ink);
      cursor: pointer;
    }
    button.primary {
      border-color: var(--accent-dark);
      background: var(--accent);
      color: #fff;
    }
    button:hover { filter: brightness(.98); }
    .metric {
      display: grid;
      grid-template-columns: 120px 1fr;
      gap: 8px;
      padding: 6px 0;
      border-top: 1px solid #eef1f5;
    }
    .metric:first-of-type { border-top: 0; }
    .metric span:first-child { color: var(--muted); }
    .path, .message {
      color: var(--muted);
      overflow-wrap: anywhere;
      font-size: 12px;
    }
    .ok { color: var(--ok); }
    .danger { color: var(--danger); }
    @media (max-width: 1050px) {
      main { grid-template-columns: 1fr; height: auto; }
      .viewer { min-height: 55vh; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Live Reference Alignment</h1>
    <div class="status" id="topStatus">Starting server UI...</div>
  </header>
  <main>
    <div class="viewer">
      <img id="stream" src="/stream" alt="Live alignment stream">
    </div>
    <aside class="sidebar">
      <section>
        <h2>Reference Image</h2>
        <form id="uploadForm" class="grid">
          <input class="wide" type="file" name="reference" accept="image/*">
          <button class="primary" type="submit">Load Image</button>
          <button type="button" id="reloadReference">Reload</button>
        </form>
        <div class="path" id="referencePath">No reference loaded.</div>
      </section>

      <section>
        <h2>Camera</h2>
        <div class="grid">
          <label>Index <input id="camera" type="number" min="0" max="9" value="0"></label>
          <label>FPS <input id="fps" type="number" min="1" max="120" value="30"></label>
          <label>Width <input id="captureWidth" type="number" min="160" value="640"></label>
          <label>Height <input id="captureHeight" type="number" min="120" value="480"></label>
          <button class="primary" id="startCamera">Start Camera</button>
          <button id="stopCamera">Stop Camera</button>
        </div>
      </section>

      <section>
        <h2>View and Overlay</h2>
        <div class="grid">
          <label class="check"><input id="overlayMode" type="checkbox" checked> Overlay mode</label>
          <label class="wide">Overlay opacity <input id="alpha" type="range" min="0" max="1" step="0.01" value="0.50"></label>
        </div>
      </section>

      <section>
        <h2>Save / Output</h2>
        <div class="grid">
          <label class="wide">Output directory <input id="outputDir" type="text" value="" placeholder="Desktop/output_images"></label>
          <button class="primary" id="saveView">Save Alignment View</button>
          <button class="primary" id="saveLive">Save Live Frame</button>
        </div>
        <div class="message" id="saveMessage"></div>
      </section>
    </aside>
  </main>
  <script>
    const ids = [
      "camera", "fps", "captureWidth", "captureHeight", "overlayMode", "alpha", "outputDir"
    ];

    function value(id) {
      const el = document.getElementById(id);
      if (el.type === "checkbox") return el.checked;
      if (el.type === "number" || el.type === "range") return Number(el.value);
      return el.value;
    }

    async function postJSON(path, payload = {}) {
      const response = await fetch(path, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
      });
      return response.json();
    }

    function collectControls() {
      const data = {};
      ids.forEach(id => data[id] = value(id));
      data.showText = false;
      data.matchDebugMode = false;
      data.matchMode = "orb";
      data.scale = 100;
      data.xOffset = 0;
      data.yOffset = 0;
      return data;
    }

    let pendingControl = null;
    function sendControlsSoon() {
      clearTimeout(pendingControl);
      pendingControl = setTimeout(() => postJSON("/api/control", collectControls()), 80);
    }

    ids.forEach(id => document.getElementById(id).addEventListener("input", sendControlsSoon));

    document.getElementById("startCamera").onclick = async () => postJSON("/api/start", collectControls());
    document.getElementById("stopCamera").onclick = async () => postJSON("/api/stop");
    document.getElementById("saveView").onclick = async () => {
      const data = await postJSON("/api/save-view", collectControls());
      document.getElementById("saveMessage").textContent = data.message || "";
    };
    document.getElementById("saveLive").onclick = async () => {
      const data = await postJSON("/api/save-live", collectControls());
      document.getElementById("saveMessage").textContent = data.message || "";
    };
    document.getElementById("reloadReference").onclick = async () => postJSON("/api/reload-reference");

    document.getElementById("uploadForm").addEventListener("submit", async event => {
      event.preventDefault();
      const formData = new FormData(event.target);
      const response = await fetch("/api/reference", {method: "POST", body: formData});
      const data = await response.json();
      if (data.message) document.getElementById("saveMessage").textContent = data.message;
    });

    document.addEventListener("keydown", event => {
      const tag = event.target.tagName.toLowerCase();
      if (tag === "input" || tag === "select") return;
      if (event.key === "o") document.getElementById("overlayMode").click();
      if (event.key === "s") document.getElementById("saveView").click();
      if (event.key === "q" || event.key === "Escape") postJSON("/api/stop");
    });

    async function pollState() {
      try {
        const response = await fetch("/api/state");
        const state = await response.json();
        document.getElementById("topStatus").textContent = state.camera_status;
        document.getElementById("referencePath").textContent = state.reference_path || "No reference loaded.";
      } catch (error) {
        document.getElementById("topStatus").textContent = "Server connection lost.";
      }
      setTimeout(pollState, 600);
    }
    pollState();
  </script>
</body>
</html>
"""


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Live reference alignment localhost GUI."
    )
    parser.add_argument("reference_image", nargs="?", help="Optional path to the saved reference image.")
    parser.add_argument("--host", default="127.0.0.1", help="Localhost server host. Default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=8501, help="Localhost server port. Default: 8501")
    parser.add_argument("--opencv-window", action="store_true", help="Use the original OpenCV window interface.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index. Default: 0")
    parser.add_argument("--mirror", action="store_true", help="Mirror live video horizontally.")
    parser.add_argument("--preview-width", type=int, default=960, help="UI preview width. Default: 960")
    parser.add_argument("--capture-width", type=int, default=640, help="Camera capture width. Default: 640")
    parser.add_argument("--capture-height", type=int, default=480, help="Camera capture height. Default: 480")
    parser.add_argument("--fps", type=int, default=30, help="Requested camera FPS. Default: 30")
    parser.add_argument(
        "--fit-mode",
        choices=["contain", "cover"],
        default="cover",
        help="Overlay fit mode without stretching. Default: cover",
    )
    parser.add_argument("--orb-features", type=int, default=1200, help="Maximum ORB features. Default: 1200")
    parser.add_argument("--max-matches", type=int, default=DEFAULT_MAX_MATCHES, help="Maximum ORB matches to draw.")
    parser.add_argument("--match-mode", choices=["orb", "markers"], default="orb", help="Debug matching mode.")
    parser.add_argument("--max-markers", type=int, default=DEFAULT_MAX_MARKERS, help="Maximum marker blobs.")
    parser.add_argument("--marker-detect-scale", type=float, default=DEFAULT_MARKER_DETECT_SCALE)
    parser.add_argument("--visual-score-scale", type=float, default=DEFAULT_VISUAL_SCORE_SCALE)
    parser.add_argument("--score-every", type=int, default=8, help="Run expensive scoring every N frames. Default: 8")
    parser.add_argument("--select-roi", action="store_true", help="Original OpenCV UI only: select a reference ROI.")
    parser.add_argument("--draw-mask", action="store_true", help="Original OpenCV UI only: draw a reference mask.")
    parser.add_argument("--skin-mask", action="store_true", help="Start with rough skin-color masking enabled.")
    parser.add_argument("--no-auto-target-mask", action="store_true", help="Disable automatic largest target mask.")
    parser.add_argument("--output-dir", default="output", help="Folder for saved frames.")
    return parser.parse_args()


def print_instructions(args):
    print("Live reference alignment prototype started.")
    print(f"Reference image: {args.reference_image}")
    print("This prototype is manual/visual only; automatic matching comes later.")
    print("Keys:")
    print("  o = toggle side-by-side / overlay")
    print("  m = toggle matching debug view")
    print("  t = toggle status text")
    print("  b = toggle ORB / marker-dot matching debug mode")
    print("  k = toggle rough skin-color mask for feature matching")
    print("  a = toggle automatic largest skin/body target mask")
    print("  s = save current alignment view and live frame")
    print("  r = reset overlay sliders")
    print("  q = quit")


class AlignmentWebState:
    """Shared state for the localhost GUI and the OpenCV processing helpers."""

    def __init__(self, args):
        self.args = args
        self.lock = threading.RLock()
        self.reference = None
        self.reference_path = None
        self.reference_feature_mask = None
        self.reference_keypoints = None
        self.reference_descriptors = None
        self.reference_markers = []
        self.orb = create_orb_detector(args.orb_features)
        self.camera = None
        self.camera_running = False
        self.camera_status = "stopped"
        self.last_display = None
        self.last_live_frame = None
        self.last_summary = None
        self.last_visual_summary = None
        self.last_error = ""
        self.frame_count = 0

        self.overlay_mode = True
        self.match_debug_mode = False
        self.show_text = False
        self.match_mode = args.match_mode
        self.alpha = 0.50
        self.scale = 100
        self.x_offset = 0
        self.y_offset = 0
        self.camera_index = args.camera
        self.capture_width = args.capture_width
        self.capture_height = args.capture_height
        self.fps = args.fps
        self.output_dir = args.output_dir
        self.skin_mask_enabled = args.skin_mask
        self.auto_target_enabled = not args.no_auto_target_mask

        if args.reference_image:
            self.load_reference_path(args.reference_image)

    def load_reference_path(self, path):
        with self.lock:
            reference = load_reference_image(path)
            reference = resize_reference_for_preview(reference, self.args.preview_width)
            self.set_reference(reference, str(path))

    def load_reference_bytes(self, data, filename):
        with self.lock:
            array = np.frombuffer(data, dtype=np.uint8)
            reference = cv2.imdecode(array, cv2.IMREAD_COLOR)
            if reference is None:
                raise ValueError("OpenCV could not decode the uploaded reference image.")
            reference = resize_reference_for_preview(reference, self.args.preview_width)
            self.set_reference(reference, filename)

    def set_reference(self, reference, label):
        self.reference = reference
        self.reference_path = label
        self.reference_feature_mask = build_reference_feature_mask(
            reference,
            None,
            skin_enabled=self.skin_mask_enabled,
            auto_target_enabled=self.auto_target_enabled,
        )
        self.reference_keypoints, self.reference_descriptors = detect_orb_features(
            self.orb,
            reference,
            self.reference_feature_mask,
        )
        self.reference_markers, _marker_mask = detect_marker_blobs(
            reference,
            self.reference_feature_mask,
            self.args.max_markers,
            self.args.marker_detect_scale,
        )
        self.last_summary = None
        self.last_visual_summary = None
        self.last_error = ""

    def start_camera(self):
        with self.lock:
            if self.camera_running:
                return
            camera = cv2.VideoCapture(self.camera_index)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)
            camera.set(cv2.CAP_PROP_FPS, self.fps)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not camera.isOpened():
                camera.release()
                self.camera = None
                self.camera_running = False
                self.camera_status = "camera not found"
                return

            self.camera = camera
            self.camera_running = True
            self.camera_status = "running"
            self.frame_count = 0

    def stop_camera(self):
        with self.lock:
            self.camera_running = False
            if self.camera is not None:
                self.camera.release()
                self.camera = None
            self.camera_status = "stopped"

    def apply_controls(self, data):
        with self.lock:
            self.camera_index = int(data.get("camera", self.camera_index))
            self.capture_width = int(data.get("captureWidth", self.capture_width))
            self.capture_height = int(data.get("captureHeight", self.capture_height))
            self.fps = int(data.get("fps", self.fps))
            self.overlay_mode = bool(data.get("overlayMode", self.overlay_mode))
            self.match_debug_mode = bool(data.get("matchDebugMode", self.match_debug_mode))
            self.show_text = bool(data.get("showText", self.show_text))
            self.match_mode = data.get("matchMode", self.match_mode)
            self.alpha = float(data.get("alpha", self.alpha))
            self.scale = int(data.get("scale", self.scale))
            self.x_offset = int(data.get("xOffset", self.x_offset))
            self.y_offset = int(data.get("yOffset", self.y_offset))
            self.output_dir = data.get("outputDir", self.output_dir)

    def reset_alignment(self):
        with self.lock:
            self.alpha = 0.50
            self.scale = 100
            self.x_offset = 0
            self.y_offset = 0

    def read_frame(self):
        if not self.camera_running or self.camera is None:
            return None
        ok, frame = self.camera.read()
        if not ok:
            self.camera_status = "camera error"
            self.stop_camera()
            return None
        if self.args.mirror:
            frame = cv2.flip(frame, 1)
        self.last_live_frame = frame.copy()
        return frame

    def render_display(self):
        with self.lock:
            if self.reference is None:
                frame = self.read_frame()
                if frame is None:
                    return self.placeholder_image("Load a reference image to begin.")
                self.last_summary = None
                self.last_visual_summary = None
                self.last_display = frame.copy()
                return frame

            frame = self.read_frame()
            if frame is None:
                display = draw_crosshair(self.reference)
                self.last_display = display.copy()
                return display

            self.frame_count += 1
            reference_height, reference_width = self.reference.shape[:2]
            live_scaled = scale_live_for_overlay(
                frame,
                reference_width,
                reference_height,
                self.scale,
                self.args.fit_mode,
            )
            live_canvas, live_mask = place_live_on_reference_canvas(
                live_scaled,
                self.reference.shape,
                self.x_offset,
                self.y_offset,
            )
            live_feature_mask = build_feature_mask(
                live_canvas,
                base_mask=live_mask,
                skin_enabled=self.skin_mask_enabled,
                auto_target_enabled=self.auto_target_enabled,
            )

            if self.match_debug_mode:
                display = self.render_debug_view(live_canvas, live_mask, live_feature_mask)
                mode_name = "matches"
            elif self.overlay_mode:
                display = blend_overlay(self.reference, live_canvas, live_mask, self.alpha)
                display = draw_crosshair(display)
                mode_name = "overlay"
            else:
                live_panel = prepare_side_by_side_live_frame(frame, self.reference.shape)
                reference_panel = draw_crosshair(self.reference)
                live_panel = draw_crosshair(live_panel)
                display = np.hstack([reference_panel, live_panel])
                mode_name = "side-by-side"

            visual_summary = self.last_visual_summary
            alignment_summary = self.last_summary
            should_score = (
                self.overlay_mode
                and not self.match_debug_mode
                and self.frame_count % max(1, self.args.score_every) == 0
            )
            if should_score:
                visual_summary = compute_visual_overlay_summary(
                    self.reference,
                    live_canvas,
                    live_mask,
                    self.reference_feature_mask,
                    self.args.visual_score_scale,
                )
                alignment_summary = self.compute_alignment_summary(live_canvas, live_mask, live_feature_mask)
                self.last_visual_summary = visual_summary
                self.last_summary = alignment_summary

            if self.show_text and not self.match_debug_mode:
                display = draw_status_text(
                    display,
                    mode_name,
                    self.alpha,
                    self.scale,
                    self.x_offset,
                    self.y_offset,
                    alignment_summary,
                    visual_summary,
                )

            self.last_display = display.copy()
            return display

    def render_debug_view(self, live_canvas, live_mask, live_feature_mask):
        mask_label = "auto target" if self.auto_target_enabled else "full image"
        if self.match_mode == "markers":
            return draw_marker_debug_view(
                self.reference,
                live_canvas,
                self.reference_feature_mask,
                live_mask,
                self.reference_markers,
                self.args.max_matches,
                mask_label,
                self.args.max_markers,
                self.args.marker_detect_scale,
            )
        return draw_match_debug_view(
            self.reference,
            live_canvas,
            self.reference_feature_mask,
            live_feature_mask,
            self.orb,
            self.reference_keypoints,
            self.reference_descriptors,
            self.args.max_matches,
            mask_label,
        )

    def compute_alignment_summary(self, live_canvas, live_mask, live_feature_mask):
        if self.match_mode == "markers":
            return compute_marker_alignment_summary(
                live_canvas,
                live_mask,
                self.reference_markers,
                self.args.max_markers,
                self.args.marker_detect_scale,
            )
        return compute_orb_alignment_summary(
            live_canvas,
            live_feature_mask,
            self.orb,
            self.reference_keypoints,
            self.reference_descriptors,
        )

    def placeholder_image(self, message):
        image = np.zeros((540, 960, 3), dtype=np.uint8)
        image[:, :] = (30, 38, 48)
        lines = [message, "Use the Reference Image panel to load an image."]
        y = 245
        for line in lines:
            cv2.putText(image, line, (36, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (240, 244, 248), 2, cv2.LINE_AA)
            y += 42
        return image

    def jpeg_bytes(self):
        image = self.render_display()
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            raise ValueError("Could not encode display frame.")
        return encoded.tobytes()

    def save_view(self):
        with self.lock:
            if self.last_display is None:
                return "No alignment view is available to save."
            output_path = resolve_output_dir(self.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = output_path / f"alignment_view_{timestamp}.jpg"
            cv2.imwrite(str(path), self.last_display)
            return f"Saved alignment view: {path}"

    def save_live(self):
        with self.lock:
            if self.last_live_frame is None:
                return "No live frame is available to save."
            output_path = resolve_output_dir(self.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = output_path / f"aligned_live_frame_{timestamp}.jpg"
            cv2.imwrite(str(path), self.last_live_frame)
            return f"Saved live frame: {path}"

    def state_payload(self):
        with self.lock:
            summary = self.last_summary
            if self.reference is None:
                score = "--"
                scale_text = "--"
                rotation_text = "--"
                guidance = "Camera preview only. Load a reference image to begin alignment."
                alignment_status = "waiting for reference"
            elif summary is None:
                score = "--"
                scale_text = "--"
                rotation_text = "--"
                guidance = "Waiting for live alignment data."
                alignment_status = "target not found"
            else:
                score = f"{summary.get('score', 0)}/100"
                scale = summary.get("scale")
                rotation = summary.get("rotation")
                scale_text = "--" if scale is None else f"{scale:.2f}"
                rotation_text = "--" if rotation is None else f"{rotation:.1f} deg"
                guidance = summary.get("hint", "No guidance available.")
                alignment_status = self.research_status(summary)

            return {
                "camera_status": self.camera_status,
                "reference_path": self.reference_path,
                "score": score,
                "scale": scale_text,
                "rotation": rotation_text,
                "guidance": guidance,
                "alignment_status": alignment_status,
                "message": self.last_error,
            }

    def research_status(self, summary):
        status = str(summary.get("status", "")).lower()
        score = int(summary.get("score", 0))
        if "target not found" in status or score == 0:
            return "target not found"
        if score >= 75:
            return "aligned"
        if score >= 55:
            return "partially aligned"
        return "unreliable match"


class AlignmentRequestHandler(BaseHTTPRequestHandler):
    """HTTP endpoints for the browser GUI."""

    server_version = "LiveReferenceAlignmentHTTP/1.0"

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_html(WEB_APP_HTML)
        elif parsed.path == "/stream":
            self.send_stream()
        elif parsed.path == "/api/state":
            self.send_json(self.server.state.state_payload())
        else:
            self.send_error(404, "Not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/control":
                data = self.read_json()
                self.server.state.apply_controls(data)
                self.send_json({"ok": True})
            elif parsed.path == "/api/start":
                data = self.read_json()
                self.server.state.apply_controls(data)
                self.server.state.start_camera()
                self.send_json({"ok": True})
            elif parsed.path == "/api/stop":
                self.server.state.stop_camera()
                self.send_json({"ok": True})
            elif parsed.path == "/api/reset":
                data = self.read_json()
                self.server.state.apply_controls(data)
                self.server.state.reset_alignment()
                self.send_json({"ok": True})
            elif parsed.path == "/api/reference":
                self.handle_reference_upload()
            elif parsed.path == "/api/reload-reference":
                reference_path = self.server.state.reference_path
                if reference_path and Path(reference_path).exists():
                    self.server.state.load_reference_path(reference_path)
                    self.send_json({"ok": True, "message": f"Reloaded {reference_path}"})
                else:
                    self.send_json({"ok": False, "message": "No disk-backed reference path is available."})
            elif parsed.path == "/api/save-view":
                data = self.read_json()
                self.server.state.apply_controls(data)
                self.send_json({"ok": True, "message": self.server.state.save_view()})
            elif parsed.path == "/api/save-live":
                data = self.read_json()
                self.server.state.apply_controls(data)
                self.send_json({"ok": True, "message": self.server.state.save_live()})
            else:
                self.send_error(404, "Not found")
        except Exception as exc:
            self.server.state.last_error = str(exc)
            self.send_json({"ok": False, "message": str(exc)}, status=500)

    def read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        body = self.rfile.read(length).decode("utf-8")
        return json.loads(body)

    def handle_reference_upload(self):
        content_type = self.headers.get("Content-Type", "")
        if content_type.startswith("multipart/form-data"):
            form = FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": content_type,
                },
            )
            field = form["reference"] if "reference" in form else None
            if field is None or not field.filename:
                self.send_json({"ok": False, "message": "No reference image was uploaded."}, status=400)
                return
            self.server.state.load_reference_bytes(field.file.read(), field.filename)
            self.send_json({"ok": True, "message": f"Loaded reference image: {field.filename}"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        data = parse_qs(self.rfile.read(length).decode("utf-8"))
        path = data.get("path", [""])[0]
        if not path:
            self.send_json({"ok": False, "message": "No reference path was provided."}, status=400)
            return
        self.server.state.load_reference_path(path)
        self.send_json({"ok": True, "message": f"Loaded reference image: {path}"})

    def send_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        while True:
            try:
                frame = self.server.state.jpeg_bytes()
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                time.sleep(1.0 / max(1, self.server.state.fps))
            except (BrokenPipeError, ConnectionResetError):
                break

    def send_html(self, html):
        data = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload, status=200):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format_text, *args):
        return


class AlignmentHTTPServer(ThreadingHTTPServer):
    def __init__(self, address, handler_class, state):
        super().__init__(address, handler_class)
        self.state = state


def run_web_gui(args):
    state = AlignmentWebState(args)
    server = AlignmentHTTPServer((args.host, args.port), AlignmentRequestHandler, state)
    url = f"http://{args.host}:{args.port}"
    print("Live reference alignment localhost GUI started.")
    print(f"Open this URL in your browser: {url}")
    print("Press Control+C in this terminal to stop the server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        state.stop_camera()
        server.server_close()


def run_opencv_window(args):
    if not args.reference_image:
        raise ValueError("The original OpenCV window mode requires a reference_image path.")
    reference = load_reference_image(args.reference_image)
    reference = resize_reference_for_preview(reference, args.preview_width)
    reference_height, reference_width = reference.shape[:2]
    orb = create_orb_detector(args.orb_features)

    reference_roi = select_reference_roi(reference) if args.select_roi else None
    reference_roi_mask = create_rect_mask(reference.shape, reference_roi) if reference_roi is not None else None
    reference_draw_mask = select_reference_draw_mask(reference) if args.draw_mask else None
    reference_manual_mask = combine_masks(reference_roi_mask, reference_draw_mask)
    auto_target_enabled = not args.no_auto_target_mask
    skin_mask_enabled = args.skin_mask
    reference_feature_mask = build_reference_feature_mask(
        reference,
        reference_manual_mask,
        skin_enabled=skin_mask_enabled,
        auto_target_enabled=auto_target_enabled,
    )
    reference_keypoints, reference_descriptors = detect_orb_features(
        orb,
        reference,
        reference_feature_mask,
    )
    reference_markers, _reference_marker_mask = detect_marker_blobs(
        reference,
        reference_feature_mask,
        args.max_markers,
        args.marker_detect_scale,
    )

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        print(f"Could not open camera index {args.camera}")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.capture_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.capture_height)
    camera.set(cv2.CAP_PROP_FPS, args.fps)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, reference_width, min(reference_height, 900))
    cv2.createTrackbar(ALPHA_TRACKBAR, WINDOW_NAME, 50, 100, nothing)
    cv2.createTrackbar(SCALE_TRACKBAR, WINDOW_NAME, 100, 200, nothing)
    cv2.createTrackbar(X_TRACKBAR, WINDOW_NAME, reference_width, reference_width * 2, nothing)
    cv2.createTrackbar(Y_TRACKBAR, WINDOW_NAME, reference_height, reference_height * 2, nothing)

    print_instructions(args)
    overlay_mode = True
    match_debug_mode = False
    show_text = True
    match_mode = args.match_mode
    last_display = reference.copy()
    last_live_frame = None

    while True:
        ok, frame = camera.read()

        if not ok:
            print("Could not read from camera.")
            break

        if args.mirror:
            frame = cv2.flip(frame, 1)

        last_live_frame = frame.copy()
        alpha = cv2.getTrackbarPos(ALPHA_TRACKBAR, WINDOW_NAME) / 100.0
        scale = cv2.getTrackbarPos(SCALE_TRACKBAR, WINDOW_NAME)
        x_track = cv2.getTrackbarPos(X_TRACKBAR, WINDOW_NAME)
        y_track = cv2.getTrackbarPos(Y_TRACKBAR, WINDOW_NAME)
        x_offset = x_track - reference_width
        y_offset = y_track - reference_height

        live_scaled = scale_live_for_overlay(
            frame,
            reference_width,
            reference_height,
            scale,
            args.fit_mode,
        )
        live_canvas, live_mask = place_live_on_reference_canvas(
            live_scaled,
            reference.shape,
            x_offset,
            y_offset,
        )
        live_feature_mask = build_feature_mask(
            live_canvas,
            base_mask=live_mask,
            skin_enabled=skin_mask_enabled,
            auto_target_enabled=auto_target_enabled,
        )

        if match_debug_mode:
            mask_parts = []
            if reference_roi is not None:
                mask_parts.append("reference ROI")
            if reference_draw_mask is not None:
                mask_parts.append("drawn mask")
            if match_mode == "markers":
                mask_parts.append("live marker blobs")
            elif reference_manual_mask is None and auto_target_enabled:
                mask_parts.append("auto target")
            elif reference_manual_mask is not None and auto_target_enabled:
                mask_parts.append("live auto target")
            if match_mode != "markers" and reference_manual_mask is None and skin_mask_enabled:
                mask_parts.append("skin mask")
            elif match_mode != "markers" and reference_manual_mask is not None and skin_mask_enabled:
                mask_parts.append("live skin mask")
            if not mask_parts:
                mask_parts.append("full image")

            if match_mode == "markers":
                display = draw_marker_debug_view(
                    reference,
                    live_canvas,
                    reference_feature_mask,
                    live_mask,
                    reference_markers,
                    args.max_matches,
                    " + ".join(mask_parts),
                    args.max_markers,
                    args.marker_detect_scale,
                )
            else:
                display = draw_match_debug_view(
                    reference,
                    live_canvas,
                    reference_feature_mask,
                    live_feature_mask,
                    orb,
                    reference_keypoints,
                    reference_descriptors,
                    args.max_matches,
                    " + ".join(mask_parts),
                )
            mode_name = "matches"
        elif overlay_mode:
            display = blend_overlay(reference, live_canvas, live_mask, alpha)
            display = draw_crosshair(display)
            mode_name = "overlay"
        else:
            live_panel = prepare_side_by_side_live_frame(frame, reference.shape)
            reference_panel = draw_crosshair(reference)
            live_panel = draw_crosshair(live_panel)
            display = np.hstack([reference_panel, live_panel])
            mode_name = "side-by-side"

        alignment_summary = None
        visual_summary = None
        if not match_debug_mode and overlay_mode:
            visual_summary = compute_visual_overlay_summary(
                reference,
                live_canvas,
                live_mask,
                reference_feature_mask,
                args.visual_score_scale,
            )
            if match_mode == "markers":
                alignment_summary = compute_marker_alignment_summary(
                    live_canvas,
                    live_mask,
                    reference_markers,
                    args.max_markers,
                    args.marker_detect_scale,
                )
            else:
                alignment_summary = compute_orb_alignment_summary(
                    live_canvas,
                    live_feature_mask,
                    orb,
                    reference_keypoints,
                    reference_descriptors,
                )

        if show_text and not match_debug_mode:
            display = draw_status_text(
                display,
                mode_name,
                alpha,
                scale,
                x_offset,
                y_offset,
                alignment_summary,
                visual_summary,
            )
        last_display = display.copy()
        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("o"):
            overlay_mode = not overlay_mode
            match_debug_mode = False
        if key == ord("m"):
            match_debug_mode = not match_debug_mode
        if key == ord("t"):
            show_text = not show_text
            print(f"Status text: {'ON' if show_text else 'OFF'}")
        if key == ord("b"):
            match_mode = "markers" if match_mode == "orb" else "orb"
            match_debug_mode = True
            print(f"Matching debug mode: {match_mode}")
        if key == ord("k"):
            skin_mask_enabled = not skin_mask_enabled
            reference_feature_mask = build_reference_feature_mask(
                reference,
                reference_manual_mask,
                skin_enabled=skin_mask_enabled,
                auto_target_enabled=auto_target_enabled,
            )
            reference_keypoints, reference_descriptors = detect_orb_features(
                orb,
                reference,
                reference_feature_mask,
            )
            reference_markers, _reference_marker_mask = detect_marker_blobs(
                reference,
                reference_feature_mask,
                args.max_markers,
                args.marker_detect_scale,
            )
            print(f"Skin-color feature mask: {'ON' if skin_mask_enabled else 'OFF'}")
        if key == ord("a"):
            auto_target_enabled = not auto_target_enabled
            reference_feature_mask = build_reference_feature_mask(
                reference,
                reference_manual_mask,
                skin_enabled=skin_mask_enabled,
                auto_target_enabled=auto_target_enabled,
            )
            reference_keypoints, reference_descriptors = detect_orb_features(
                orb,
                reference,
                reference_feature_mask,
            )
            reference_markers, _reference_marker_mask = detect_marker_blobs(
                reference,
                reference_feature_mask,
                args.max_markers,
                args.marker_detect_scale,
            )
            print(f"Automatic target mask: {'ON' if auto_target_enabled else 'OFF'}")
        if key == ord("s") and last_live_frame is not None:
            save_alignment_outputs(last_display, last_live_frame, args.output_dir)
        if key == ord("r"):
            cv2.setTrackbarPos(ALPHA_TRACKBAR, WINDOW_NAME, 50)
            cv2.setTrackbarPos(SCALE_TRACKBAR, WINDOW_NAME, 100)
            cv2.setTrackbarPos(X_TRACKBAR, WINDOW_NAME, reference_width)
            cv2.setTrackbarPos(Y_TRACKBAR, WINDOW_NAME, reference_height)

    camera.release()
    cv2.destroyAllWindows()


def main():
    args = parse_arguments()
    if args.opencv_window:
        run_opencv_window(args)
    else:
        run_web_gui(args)


if __name__ == "__main__":
    main()
