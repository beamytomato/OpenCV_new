import argparse
import base64
import errno
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import sys
import threading
import traceback
from types import SimpleNamespace
import time
from datetime import datetime
from pathlib import Path
import urllib.parse
import uuid
import webbrowser

import cv2
import live_reference_alignment as alignment_gui
from local_form import FieldStorage
import numpy as np


# -----------------------------
# TUNING PARAMETERS
# -----------------------------
# These values are intentionally near the top so you can tune the detector.

# Silver/metal mask settings in HSV.
# Silver metal is usually low saturation with medium/high brightness.
LOWER_SILVER = np.array([0, 0, 70], dtype=np.uint8)
UPPER_SILVER = np.array([179, 95, 255], dtype=np.uint8)

# Extra gray-like BGR filter. Metal usually has similar B, G, and R values.
MAX_BGR_CHANNEL_DIFFERENCE = 65
STRICT_MAX_BGR_CHANNEL_DIFFERENCE = 45
STRICT_MIN_VALUE = 90
MIN_HOUGH_METAL_FRACTION = 0.08
MIN_HOUGH_EDGE_FRACTION = 0.04
IGNORE_TOP_FRACTION = 0.22
MIN_EDGE_CONTOUR_AREA = 900
MAX_EDGE_CONTOUR_ASPECT_RATIO = 1.8

# Shape/size filters for the outer ring.
MIN_RING_AREA = 800
MAX_RING_AREA_FRACTION = 0.45
MIN_MAJOR_AXIS = 40
MAX_ASPECT_RATIO = 1.8

# A real ring should have a hole. These limits compare the inner hole area to
# the outer contour area. Loosen these if the hole is dark/shadowed or occluded.
MIN_HOLE_RATIO = 0.05
MAX_HOLE_RATIO = 0.65

# Hough circle fallback settings.
# These are used when the metal color/contour detector fails.
HOUGH_DP = 1.2
HOUGH_MIN_DIST_FRACTION = 0.18
HOUGH_PARAM1 = 120
HOUGH_PARAM2 = 28
HOUGH_MIN_RADIUS_FRACTION = 0.035
HOUGH_MAX_RADIUS_FRACTION = 0.25

# Annotation settings.
CROSSHAIR_LENGTH = 55
MIN_OUTLINE_THICKNESS = 4
MAX_OUTLINE_THICKNESS = 10

# Dark visual landmark settings.
# These are non-diagnostic visual landmarks only. They are meant to mark clear,
# dark, mole-like spots that may help with image alignment later.
MAX_MOLE_LANDMARKS = 7
MIN_MOLE_AREA = 10
MAX_MOLE_AREA = 1800
MOLE_DARK_RESPONSE_THRESHOLD = 6
MOLE_COLOR_RESPONSE_THRESHOLD = 7
MOLE_LARGE_PATCH_THRESHOLD = 5
MOLE_COLOR_RESPONSE_WEIGHT = 0.75
MOLE_LARGE_PATCH_WEIGHT = 0.35
MOLE_MIN_CONTRAST = 4.5
MOLE_MAX_ASPECT_RATIO = 6.0
MOLE_MIN_CIRCULARITY = 0.05
MOLE_RING_EXCLUSION_SCALE = 1.05
MOLE_BOX_PADDING = 8
MOLE_MIN_DISTANCE = 78
MIN_SKIN_COMPONENT_AREA_FRACTION = 0.015
MOLE_SKIN_ERODE_KERNEL_SIZE = 27
MIN_MOLE_SKIN_COVERAGE = 0.90
MIN_MOLE_INNER_SKIN_COVERAGE = 0.35
MIN_MOLE_STRICT_SKIN_COVERAGE = 0.55
MIN_MOLE_MEAN_VALUE = 75
MIN_MOLE_SKIN_EDGE_DISTANCE = 18
MOLE_IMAGE_EDGE_MARGIN = 28
MOLE_NMS_EXPAND_PIXELS = 28
RING_CENTER_SKIN_SEED_RADIUS = 90
MOLE_EDGE_CONTEXT_PADDING = 55
MAX_MOLE_CONTEXT_NON_SKIN_FRACTION = 0.28
MOLE_ADAPTIVE_PERCENTILE = 96.5
MOLE_STRONG_CORE_PERCENTILE = 98.5
MOLE_MIN_CORE_FRACTION = 0.035
MOLE_MIN_CORE_PIXELS = 4
MOLE_BOX_COLORS = [
    (255, 0, 255),   # magenta
    (255, 255, 0),   # cyan
    (0, 165, 255),   # orange
    (0, 255, 255),   # yellow
    (0, 255, 0),     # green
    (255, 0, 0),     # blue
    (255, 255, 255), # white
]

MIN_MANUAL_LANDMARK_BOX_SIZE = 4


def load_image(image_path):
    """Load an image from disk and stop with a clear message if it fails."""
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    return image


def capture_webcam_photo(camera_index=0, capture_path=None):
    """Open a webcam preview and capture one frame when the user presses c."""
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    if capture_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        capture_path = output_dir / f"webcam_capture_{timestamp}.jpg"
    else:
        capture_path = Path(capture_path)
        capture_path.parent.mkdir(parents=True, exist_ok=True)

    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open webcam index {camera_index}. "
            "Try --camera-index 1 if you have more than one camera."
        )

    window_name = "Webcam Capture - press c/space to capture, q/esc to cancel"
    captured_frame = None

    try:
        print("Webcam preview opened.")
        print("Press c or Space to capture. Press q or Esc to cancel.")

        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the webcam.")

            preview = frame.copy()
            cv2.putText(
                preview,
                "Press c/Space to capture, q/Esc to cancel",
                (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                preview,
                "Press c/Space to capture, q/Esc to cancel",
                (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("c"), ord(" "), 13):
                captured_frame = frame.copy()
                break
            if key in (ord("q"), 27):
                raise RuntimeError("Webcam capture cancelled.")
    finally:
        camera.release()
        cv2.destroyWindow(window_name)

    if not cv2.imwrite(str(capture_path), captured_frame):
        raise RuntimeError(f"Could not save captured webcam image: {capture_path}")

    print(f"Saved webcam capture: {capture_path}")
    return captured_frame, capture_path


def create_metal_mask(image):
    """
    Create a binary mask for silver/gray metallic regions.

    The mask combines:
    1. HSV low-saturation, medium/high-brightness filtering.
    2. BGR gray-like filtering where B/G/R channels are close together.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv, LOWER_SILVER, UPPER_SILVER)

    b, g, r = cv2.split(image)
    max_channel = np.maximum(np.maximum(b, g), r)
    min_channel = np.minimum(np.minimum(b, g), r)
    channel_difference = cv2.subtract(max_channel, min_channel)
    gray_like_mask = cv2.inRange(
        channel_difference,
        0,
        MAX_BGR_CHANNEL_DIFFERENCE,
    )

    metal_mask = cv2.bitwise_and(hsv_mask, gray_like_mask)

    # Clean noise and fill small gaps in the ring.
    kernel = np.ones((7, 7), np.uint8)
    metal_mask = cv2.morphologyEx(metal_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    metal_mask = cv2.morphologyEx(metal_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    return metal_mask


def create_strict_metal_mask(image):
    """
    Create a stricter metal mask for scoring Hough circle candidates.

    This mask is not used alone for contour detection because it can miss
    shadowed parts of the ring. It is useful for rejecting false Hough circles
    on striped fabric, skin, or background.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    b, g, r = cv2.split(image)
    max_channel = np.maximum(np.maximum(b, g), r)
    min_channel = np.minimum(np.minimum(b, g), r)
    channel_difference = cv2.subtract(max_channel, min_channel)

    low_saturation = cv2.inRange(s, 0, 65)
    bright_enough = cv2.inRange(v, STRICT_MIN_VALUE, 255)
    gray_like = cv2.inRange(channel_difference, 0, STRICT_MAX_BGR_CHANNEL_DIFFERENCE)

    strict_mask = cv2.bitwise_and(low_saturation, bright_enough)
    strict_mask = cv2.bitwise_and(strict_mask, gray_like)

    kernel = np.ones((5, 5), np.uint8)
    strict_mask = cv2.morphologyEx(strict_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    strict_mask = cv2.morphologyEx(strict_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return strict_mask


def contour_circularity(contour):
    """Return circularity. A perfect circle is near 1.0."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        return 0

    return 4 * np.pi * area / (perimeter * perimeter)


def find_largest_child_hole(contour_index, contours, hierarchy):
    """Find the largest child contour inside a candidate outer contour."""
    if hierarchy is None:
        return None, 0

    hierarchy = hierarchy[0]
    child_index = hierarchy[contour_index][2]
    largest_child = None
    largest_area = 0

    while child_index != -1:
        child_area = cv2.contourArea(contours[child_index])
        if child_area > largest_area:
            largest_area = child_area
            largest_child = contours[child_index]

        child_index = hierarchy[child_index][0]

    return largest_child, largest_area


def score_ring_candidate(area, hole_ratio, circularity, aspect_ratio):
    """Score candidates so the best ring-like object wins."""
    aspect_penalty = abs(1.0 - aspect_ratio)
    hole_score = 1.0 - abs(0.25 - hole_ratio)
    return area * max(circularity, 0.1) * max(hole_score, 0.1) / (1.0 + aspect_penalty)


def create_edge_image(image):
    """Create an edge image for shape-based ring fallback detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (7, 7), 1.5)
    edges = cv2.Canny(blurred, 50, 150)
    return gray, blurred, edges


def circle_to_ellipse(circle):
    """Convert a Hough circle into an OpenCV ellipse tuple for annotation."""
    x, y, radius = circle
    diameter = radius * 2
    return ((float(x), float(y)), (float(diameter), float(diameter)), 0.0)


def create_lower_image_mask(image_shape):
    """Mask out the top of the image where clothing/fabric often appears."""
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    y_start = int(height * IGNORE_TOP_FRACTION)
    mask[y_start:, :] = 255
    return mask


def score_edge_ellipse_candidate(contour, ellipse, strict_metal_mask, image_shape):
    """Score a closed edge contour as a possible ring outline."""
    height, width = image_shape[:2]
    (cx, cy), (ellipse_width, ellipse_height), _angle = ellipse

    if cy < height * IGNORE_TOP_FRACTION:
        return 0

    major = max(ellipse_width, ellipse_height)
    minor = min(ellipse_width, ellipse_height)

    if minor <= 0:
        return 0

    aspect_ratio = major / minor
    if aspect_ratio > MAX_EDGE_CONTOUR_ASPECT_RATIO:
        return 0

    area = cv2.contourArea(contour)
    if area < MIN_EDGE_CONTOUR_AREA:
        return 0

    max_area = height * width * MAX_RING_AREA_FRACTION
    if area > max_area:
        return 0

    circularity = contour_circularity(contour)

    ellipse_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(ellipse_mask, ellipse, 255, -1)

    inner_mask = np.zeros((height, width), dtype=np.uint8)
    inner_axes = (max(1, int(ellipse_width * 0.28)), max(1, int(ellipse_height * 0.28)))
    cv2.ellipse(
        inner_mask,
        (int(round(cx)), int(round(cy))),
        inner_axes,
        0,
        0,
        360,
        255,
        -1,
    )

    annulus_mask = cv2.subtract(ellipse_mask, inner_mask)
    annulus_area = cv2.countNonZero(annulus_mask)
    if annulus_area == 0:
        return 0

    strict_metal_pixels = cv2.countNonZero(cv2.bitwise_and(strict_metal_mask, annulus_mask))
    strict_metal_fraction = strict_metal_pixels / annulus_area

    # The real ring may have shadows, so do not require a huge metal fraction.
    # But fabric patterns usually have almost none.
    if strict_metal_fraction < 0.015:
        return 0

    size_score = np.sqrt(area)
    return size_score * max(circularity, 0.1) * (1.0 + strict_metal_fraction * 8.0) / aspect_ratio


def detect_ring_with_edge_contours(image, strict_metal_mask):
    """Fallback detector that finds round closed contours in the edge map."""
    gray, blurred, edges = create_edge_image(image)
    lower_mask = create_lower_image_mask(image.shape)
    lower_edges = cv2.bitwise_and(edges, lower_mask)

    # Join broken circular edge fragments around the metal ring.
    kernel = np.ones((5, 5), np.uint8)
    closed_edges = cv2.dilate(lower_edges, kernel, iterations=1)
    closed_edges = cv2.morphologyEx(closed_edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        closed_edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    best = None
    best_score = 0

    for contour in contours:
        if len(contour) < 5:
            continue

        ellipse = cv2.fitEllipse(contour)
        score = score_edge_ellipse_candidate(
            contour,
            ellipse,
            strict_metal_mask,
            image.shape,
        )

        if score > best_score:
            best_score = score
            best = {
                "contour": contour,
                "ellipse": ellipse,
                "hole_contour": None,
                "area": cv2.contourArea(contour),
                "hole_area": 0,
                "hole_ratio": 0,
                "circularity": contour_circularity(contour),
                "aspect_ratio": max(ellipse[1]) / max(1, min(ellipse[1])),
                "score": score,
                "method": "edge_contour_ellipse",
            }

    debug = {
        "gray": gray,
        "blurred": blurred,
        "edges": edges,
        "lower_edges": lower_edges,
        "closed_edges": closed_edges,
    }

    return best, debug


def score_hough_circle(circle, metal_mask, strict_metal_mask, edges, image_shape):
    """
    Score a possible ring circle using edge and metal evidence.

    The score favors circles with a strong edge around the circumference and
    some metal-mask evidence in the annulus, while rejecting huge background
    structures.
    """
    height, width = image_shape[:2]
    x, y, radius = circle

    if x - radius < 0 or y - radius < 0 or x + radius >= width or y + radius >= height:
        return 0

    outer = np.zeros((height, width), dtype=np.uint8)
    inner = np.zeros((height, width), dtype=np.uint8)
    circumference = np.zeros((height, width), dtype=np.uint8)

    cv2.circle(outer, (x, y), radius, 255, -1)
    cv2.circle(inner, (x, y), max(1, int(radius * 0.45)), 255, -1)
    annulus = cv2.subtract(outer, inner)
    cv2.circle(circumference, (x, y), radius, 255, 4)

    annulus_area = cv2.countNonZero(annulus)
    if annulus_area == 0:
        return 0

    metal_pixels = cv2.countNonZero(cv2.bitwise_and(metal_mask, annulus))
    strict_metal_pixels = cv2.countNonZero(cv2.bitwise_and(strict_metal_mask, annulus))
    edge_pixels = cv2.countNonZero(cv2.bitwise_and(edges, circumference))
    circumference_pixels = cv2.countNonZero(circumference)

    metal_fraction = metal_pixels / annulus_area
    strict_metal_fraction = strict_metal_pixels / annulus_area
    edge_fraction = edge_pixels / max(circumference_pixels, 1)

    if strict_metal_fraction < MIN_HOUGH_METAL_FRACTION:
        return 0

    if edge_fraction < MIN_HOUGH_EDGE_FRACTION:
        return 0

    # Hough may find inner hole circles too. Prefer outer-size circles by giving
    # a mild radius bonus, while edge evidence carries most of the score.
    radius_bonus = radius / max(width, height)
    return (edge_fraction * 5.0) + (metal_fraction * 1.0) + (strict_metal_fraction * 5.0) + radius_bonus


def detect_ring_with_hough(image, metal_mask, strict_metal_mask):
    """Fallback detector using Hough circles when contour detection fails."""
    gray, blurred, edges = create_edge_image(image)
    height, width = image.shape[:2]
    min_dimension = min(height, width)

    min_radius = max(10, int(min_dimension * HOUGH_MIN_RADIUS_FRACTION))
    max_radius = max(min_radius + 10, int(min_dimension * HOUGH_MAX_RADIUS_FRACTION))
    min_dist = max(30, int(min_dimension * HOUGH_MIN_DIST_FRACTION))

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=min_dist,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return None, {"gray": gray, "blurred": blurred, "edges": edges}

    circles = np.round(circles[0]).astype(int)
    best_circle = None
    best_score = 0

    for circle in circles:
        score = score_hough_circle(circle, metal_mask, strict_metal_mask, edges, image.shape)
        if score > best_score:
            best_score = score
            best_circle = circle

    if best_circle is None:
        return None, {"gray": gray, "blurred": blurred, "edges": edges}

    x, y, radius = best_circle
    ellipse = circle_to_ellipse((x, y, radius))
    detection = {
        "contour": None,
        "ellipse": ellipse,
        "hole_contour": None,
        "area": float(np.pi * radius * radius),
        "hole_area": 0,
        "hole_ratio": 0,
        "circularity": 1.0,
        "aspect_ratio": 1.0,
        "score": best_score,
        "method": "hough_circle",
    }

    return detection, {"gray": gray, "blurred": blurred, "edges": edges}


def detect_ring(image):
    """
    Detect the metal ring and return the best fitted ellipse plus debug data.

    The detector uses contours with hierarchy so it can prefer metal objects
    that contain an inner hole, which is the defining ring shape.
    """
    metal_mask = create_metal_mask(image)
    strict_metal_mask = create_strict_metal_mask(image)
    contours, hierarchy = cv2.findContours(
        metal_mask,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    image_height, image_width = image.shape[:2]
    image_area = image_height * image_width
    max_ring_area = image_area * MAX_RING_AREA_FRACTION

    best = None
    best_score = 0

    for index, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        if area < MIN_RING_AREA or area > max_ring_area:
            continue

        if len(contour) < 5:
            continue

        ellipse = cv2.fitEllipse(contour)
        (_cx, _cy), (width, height), _angle = ellipse
        major_axis = max(width, height)
        minor_axis = min(width, height)

        if minor_axis <= 0:
            continue

        if major_axis < MIN_MAJOR_AXIS:
            continue

        aspect_ratio = major_axis / minor_axis
        if aspect_ratio > MAX_ASPECT_RATIO:
            continue

        hole_contour, hole_area = find_largest_child_hole(index, contours, hierarchy)
        hole_ratio = hole_area / area if area > 0 else 0

        if hole_ratio < MIN_HOLE_RATIO or hole_ratio > MAX_HOLE_RATIO:
            continue

        circularity = contour_circularity(contour)
        score = score_ring_candidate(area, hole_ratio, circularity, aspect_ratio)

        if score > best_score:
            best_score = score
            best = {
                "contour": contour,
                "ellipse": ellipse,
                "hole_contour": hole_contour,
                "area": area,
                "hole_area": hole_area,
                "hole_ratio": hole_ratio,
                "circularity": circularity,
                "aspect_ratio": aspect_ratio,
                "score": score,
            }

    hough_debug = {}

    if best is None:
        best, hough_debug = detect_ring_with_edge_contours(image, strict_metal_mask)

    if best is None:
        best, hough_debug = detect_ring_with_hough(image, metal_mask, strict_metal_mask)
    elif "method" not in best:
        best["method"] = "metal_mask_contour"

    debug = {
        "metal_mask": metal_mask,
        "strict_metal_mask": strict_metal_mask,
    }
    debug.update(hough_debug)

    return best, debug


def ring_outline_thickness(ellipse):
    """Scale annotation thickness based on detected ring size."""
    (_center, axes, _angle) = ellipse
    major_axis = max(axes)
    thickness = int(round(major_axis / 70))
    return max(MIN_OUTLINE_THICKNESS, min(MAX_OUTLINE_THICKNESS, thickness))


def create_ring_exclusion_mask(image_shape, detection, scale=MOLE_RING_EXCLUSION_SCALE):
    """Create a filled mask around the detected ring so it is not boxed as a landmark."""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    if detection is None:
        return mask

    (cx, cy), (width, height), angle = detection["ellipse"]
    axes = (
        max(1, int(round(width * scale / 2))),
        max(1, int(round(height * scale / 2))),
    )
    center = (int(round(cx)), int(round(cy)))
    cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
    return mask


def landmark_center_distance(landmark_a, landmark_b):
    """Return pixel distance between two landmark centers."""
    ax, ay = landmark_a["center"]
    bx, by = landmark_b["center"]
    return float(np.hypot(ax - bx, ay - by))


def expanded_bbox(landmark, expand_pixels=MOLE_NMS_EXPAND_PIXELS):
    """Return an expanded bounding box as x1, y1, x2, y2."""
    x, y, width, height = landmark["bbox"]
    return (
        x - expand_pixels,
        y - expand_pixels,
        x + width + expand_pixels,
        y + height + expand_pixels,
    )


def bbox_overlap_ratio(landmark_a, landmark_b):
    """Return overlap ratio between two expanded landmark boxes."""
    ax1, ay1, ax2, ay2 = expanded_bbox(landmark_a)
    bx1, by1, bx2, by2 = expanded_bbox(landmark_b)

    overlap_x1 = max(ax1, bx1)
    overlap_y1 = max(ay1, by1)
    overlap_x2 = min(ax2, bx2)
    overlap_y2 = min(ay2, by2)

    overlap_width = max(0, overlap_x2 - overlap_x1)
    overlap_height = max(0, overlap_y2 - overlap_y1)
    overlap_area = overlap_width * overlap_height

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return overlap_area / min(area_a, area_b)


def nearby_selected_count(candidate, selected, radius):
    """Count selected landmarks within a radius of a candidate."""
    return sum(
        1
        for existing in selected
        if landmark_center_distance(candidate, existing) < radius
    )


def create_skin_mask(image):
    """
    Create a broad skin-region mask to keep dark landmarks on skin only.

    This is a color/shape mask for computer-vision filtering, not a medical
    skin classifier. It intentionally favors large continuous skin-colored
    regions and rejects small background objects.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    lower_skin_hsv_1 = np.array([0, 18, 35], dtype=np.uint8)
    upper_skin_hsv_1 = np.array([35, 210, 255], dtype=np.uint8)
    lower_skin_hsv_2 = np.array([160, 18, 35], dtype=np.uint8)
    upper_skin_hsv_2 = np.array([179, 210, 255], dtype=np.uint8)
    hsv_mask_1 = cv2.inRange(hsv, lower_skin_hsv_1, upper_skin_hsv_1)
    hsv_mask_2 = cv2.inRange(hsv, lower_skin_hsv_2, upper_skin_hsv_2)
    hsv_mask = cv2.bitwise_or(hsv_mask_1, hsv_mask_2)

    lower_skin_ycrcb = np.array([35, 128, 65], dtype=np.uint8)
    upper_skin_ycrcb = np.array([255, 185, 145], dtype=np.uint8)
    ycrcb_mask = cv2.inRange(ycrcb, lower_skin_ycrcb, upper_skin_ycrcb)
    _h, saturation, value = cv2.split(hsv)
    not_gray_background = cv2.inRange(saturation, 12, 255)
    bright_enough = cv2.inRange(value, 45, 255)
    ycrcb_mask = cv2.bitwise_and(ycrcb_mask, not_gray_background)
    ycrcb_mask = cv2.bitwise_and(ycrcb_mask, bright_enough)

    skin_mask = cv2.bitwise_or(hsv_mask, ycrcb_mask)

    kernel = np.ones((9, 9), np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        skin_mask,
        connectivity=8,
    )
    cleaned_mask = np.zeros_like(skin_mask)
    min_area = image.shape[0] * image.shape[1] * MIN_SKIN_COMPONENT_AREA_FRACTION

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned_mask[labels == label] = 255

    return cleaned_mask


def create_strict_skin_mask(image):
    """
    Create a stricter skin mask for rejecting background landmark candidates.

    The broad skin mask is useful for keeping enough limb area. This stricter
    version is used at the candidate level so dark fabric/background does not
    count as skin just because it touches the limb boundary.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    lower_skin_hsv_1 = np.array([0, 22, 60], dtype=np.uint8)
    upper_skin_hsv_1 = np.array([35, 185, 255], dtype=np.uint8)
    lower_skin_hsv_2 = np.array([160, 22, 60], dtype=np.uint8)
    upper_skin_hsv_2 = np.array([179, 185, 255], dtype=np.uint8)
    hsv_mask_1 = cv2.inRange(hsv, lower_skin_hsv_1, upper_skin_hsv_1)
    hsv_mask_2 = cv2.inRange(hsv, lower_skin_hsv_2, upper_skin_hsv_2)
    hsv_mask = cv2.bitwise_or(hsv_mask_1, hsv_mask_2)

    lower_skin_ycrcb = np.array([50, 132, 72], dtype=np.uint8)
    upper_skin_ycrcb = np.array([255, 182, 140], dtype=np.uint8)
    ycrcb_mask = cv2.inRange(ycrcb, lower_skin_ycrcb, upper_skin_ycrcb)

    strict_mask = cv2.bitwise_and(hsv_mask, ycrcb_mask)
    kernel = np.ones((5, 5), np.uint8)
    strict_mask = cv2.morphologyEx(strict_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    strict_mask = cv2.morphologyEx(strict_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return strict_mask


def keep_ring_connected_skin_component(skin_mask, ring_detection):
    """
    Keep the skin-mask component connected to the ring center area.

    This helps reject background objects that happen to pass the broad skin-color
    thresholds, especially fabric or shadows near the edge of the limb.
    """
    if ring_detection is None:
        return skin_mask

    (cx, cy), (_width, _height), _angle = ring_detection["ellipse"]
    center = (int(round(cx)), int(round(cy)))
    height, width = skin_mask.shape[:2]

    seed_mask = np.zeros_like(skin_mask)
    cv2.circle(seed_mask, center, RING_CENTER_SKIN_SEED_RADIUS, 255, -1)
    seed_mask = cv2.bitwise_and(seed_mask, skin_mask)

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        skin_mask,
        connectivity=8,
    )
    if num_labels <= 1:
        return skin_mask

    total_skin_area = cv2.countNonZero(skin_mask)
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

    seed_labels = labels[seed_mask > 0]
    if seed_labels.size == 0:
        # Fall back to the largest skin-colored component if the ring center
        # lands inside metal/shadow instead of visible skin.
        connected_mask = np.zeros_like(skin_mask)
        connected_mask[labels == largest_label] = 255
        return connected_mask

    label_counts = np.bincount(seed_labels)
    label_counts[0] = 0
    selected_label = int(np.argmax(label_counts))

    if selected_label <= 0:
        return skin_mask

    connected_mask = np.zeros_like(skin_mask)
    connected_mask[labels == selected_label] = 255

    selected_area = cv2.countNonZero(connected_mask)
    if selected_area < total_skin_area * 0.25:
        # The ring center often sees only the small skin patch inside the ring
        # hole. That component is not useful for finding landmarks around the
        # ring, so use the main visible skin component instead.
        connected_mask = np.zeros_like(skin_mask)
        connected_mask[labels == largest_label] = 255

    return connected_mask


def detect_mole_landmarks(image, ring_detection=None, max_landmarks=MAX_MOLE_LANDMARKS):
    """
    Detect up to max_landmarks prominent dark visual landmarks.

    This does not diagnose skin findings. It only finds small, dark, clear
    image features that may be useful as visual registration landmarks.
    """
    if max_landmarks <= 0:
        return [], {}

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    local_blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
    small_background = cv2.GaussianBlur(local_blur, (41, 41), 0)
    large_background = cv2.GaussianBlur(local_blur, (91, 91), 0)
    small_dark_response = cv2.subtract(small_background, local_blur)
    large_dark_response = cv2.subtract(large_background, local_blur)
    dark_response = cv2.max(small_dark_response, large_dark_response)
    large_patch_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81))
    large_patch_response = cv2.morphologyEx(
        local_blur,
        cv2.MORPH_BLACKHAT,
        large_patch_kernel,
    )

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    _lab_l, lab_a, lab_b = cv2.split(lab)
    lab_a_background = cv2.GaussianBlur(lab_a, (91, 91), 0)
    lab_b_background = cv2.GaussianBlur(lab_b, (91, 91), 0)
    color_response = cv2.addWeighted(
        cv2.absdiff(lab_a, lab_a_background),
        0.5,
        cv2.absdiff(lab_b, lab_b_background),
        0.5,
        0,
    )
    pigment_response = cv2.addWeighted(
        dark_response,
        1.0,
        color_response,
        MOLE_COLOR_RESPONSE_WEIGHT,
        0,
    )
    pigment_response = cv2.addWeighted(
        pigment_response,
        1.0,
        large_patch_response,
        MOLE_LARGE_PATCH_WEIGHT,
        0,
    )

    _, dark_mask = cv2.threshold(
        dark_response,
        MOLE_DARK_RESPONSE_THRESHOLD,
        255,
        cv2.THRESH_BINARY,
    )
    _, color_mask = cv2.threshold(
        color_response,
        MOLE_COLOR_RESPONSE_THRESHOLD,
        255,
        cv2.THRESH_BINARY,
    )
    _, large_patch_mask = cv2.threshold(
        large_patch_response,
        MOLE_LARGE_PATCH_THRESHOLD,
        255,
        cv2.THRESH_BINARY,
    )
    dark_mask = cv2.bitwise_or(dark_mask, color_mask)
    dark_mask = cv2.bitwise_or(dark_mask, large_patch_mask)

    raw_skin_mask = create_skin_mask(image)
    strict_skin_mask = create_strict_skin_mask(image)
    skin_mask = keep_ring_connected_skin_component(raw_skin_mask, ring_detection)
    minimum_expected_skin_area = image.shape[0] * image.shape[1] * 0.05
    if cv2.countNonZero(skin_mask) < minimum_expected_skin_area:
        skin_mask = raw_skin_mask
    skin_mask = cv2.bitwise_and(skin_mask, cv2.dilate(strict_skin_mask, np.ones((15, 15), np.uint8), iterations=1))
    skin_distance = cv2.distanceTransform(skin_mask, cv2.DIST_L2, 5)
    erode_kernel = np.ones(
        (MOLE_SKIN_ERODE_KERNEL_SIZE, MOLE_SKIN_ERODE_KERNEL_SIZE),
        np.uint8,
    )
    inner_skin_mask = cv2.erode(skin_mask, erode_kernel, iterations=1)
    ring_exclusion_mask = create_ring_exclusion_mask(image.shape, ring_detection)
    valid_region_mask = cv2.bitwise_and(inner_skin_mask, cv2.bitwise_not(ring_exclusion_mask))

    valid_pigment_values = pigment_response[valid_region_mask > 0]
    if valid_pigment_values.size > 0:
        adaptive_threshold = max(
            MOLE_MIN_CONTRAST,
            float(np.percentile(valid_pigment_values, MOLE_ADAPTIVE_PERCENTILE)),
        )
        strong_core_threshold = max(
            adaptive_threshold + 1.5,
            float(np.percentile(valid_pigment_values, MOLE_STRONG_CORE_PERCENTILE)),
        )
    else:
        adaptive_threshold = MOLE_MIN_CONTRAST
        strong_core_threshold = MOLE_MIN_CONTRAST + 3.0

    _, strong_core_mask = cv2.threshold(
        pigment_response,
        strong_core_threshold,
        255,
        cv2.THRESH_BINARY,
    )
    strong_core_mask = strong_core_mask.astype(np.uint8)

    dark_mask = cv2.bitwise_and(strong_core_mask, valid_region_mask)
    strong_core_mask = cv2.bitwise_and(strong_core_mask, valid_region_mask)

    kernel = np.ones((3, 3), np.uint8)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    merge_kernel = np.ones((5, 5), np.uint8)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, merge_kernel, iterations=1)

    contours, _ = cv2.findContours(
        dark_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_MOLE_AREA or area > MAX_MOLE_AREA:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        if width <= 0 or height <= 0:
            continue

        center_x = x + width // 2
        center_y = y + height // 2
        if (
            center_x < MOLE_IMAGE_EDGE_MARGIN
            or center_y < MOLE_IMAGE_EDGE_MARGIN
            or center_x > image.shape[1] - MOLE_IMAGE_EDGE_MARGIN
            or center_y > image.shape[0] - MOLE_IMAGE_EDGE_MARGIN
        ):
            continue
        if inner_skin_mask[center_y, center_x] == 0:
            continue
        if skin_distance[center_y, center_x] < MIN_MOLE_SKIN_EDGE_DISTANCE:
            continue

        padded_x1 = max(0, x - MOLE_BOX_PADDING)
        padded_y1 = max(0, y - MOLE_BOX_PADDING)
        padded_x2 = min(image.shape[1], x + width + MOLE_BOX_PADDING)
        padded_y2 = min(image.shape[0], y + height + MOLE_BOX_PADDING)
        padded_skin = skin_mask[padded_y1:padded_y2, padded_x1:padded_x2]
        padded_inner_skin = inner_skin_mask[padded_y1:padded_y2, padded_x1:padded_x2]
        padded_strict_skin = strict_skin_mask[padded_y1:padded_y2, padded_x1:padded_x2]
        padded_area = padded_skin.size
        skin_coverage = cv2.countNonZero(padded_skin) / max(padded_area, 1)
        inner_skin_coverage = cv2.countNonZero(padded_inner_skin) / max(padded_area, 1)
        strict_skin_coverage = cv2.countNonZero(padded_strict_skin) / max(padded_area, 1)
        if skin_coverage < MIN_MOLE_SKIN_COVERAGE:
            continue
        if inner_skin_coverage < MIN_MOLE_INNER_SKIN_COVERAGE:
            continue
        if strict_skin_coverage < MIN_MOLE_STRICT_SKIN_COVERAGE:
            continue

        context_x1 = max(0, x - MOLE_EDGE_CONTEXT_PADDING)
        context_y1 = max(0, y - MOLE_EDGE_CONTEXT_PADDING)
        context_x2 = min(image.shape[1], x + width + MOLE_EDGE_CONTEXT_PADDING)
        context_y2 = min(image.shape[0], y + height + MOLE_EDGE_CONTEXT_PADDING)
        context_skin = strict_skin_mask[context_y1:context_y2, context_x1:context_x2]
        context_area = context_skin.size
        context_non_skin_fraction = 1.0 - (
            cv2.countNonZero(context_skin) / max(context_area, 1)
        )
        if context_non_skin_fraction > MAX_MOLE_CONTEXT_NON_SKIN_FRACTION:
            continue

        aspect_ratio = max(width / height, height / width)
        if aspect_ratio > MOLE_MAX_ASPECT_RATIO:
            continue

        circularity = contour_circularity(contour)
        if circularity < MOLE_MIN_CIRCULARITY:
            continue

        contour_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
        core_pixels = cv2.countNonZero(cv2.bitwise_and(strong_core_mask, contour_mask))
        core_fraction = core_pixels / max(area, 1.0)
        if core_pixels < MOLE_MIN_CORE_PIXELS and core_fraction < MOLE_MIN_CORE_FRACTION:
            continue

        contrast = cv2.mean(dark_response, mask=contour_mask)[0]
        color_contrast = cv2.mean(color_response, mask=contour_mask)[0]
        large_patch_contrast = cv2.mean(large_patch_response, mask=contour_mask)[0]
        pigment_contrast = cv2.mean(pigment_response, mask=contour_mask)[0]
        mean_value = cv2.mean(hsv[:, :, 2], mask=contour_mask)[0]
        if mean_value < MIN_MOLE_MEAN_VALUE:
            continue
        if pigment_contrast < MOLE_MIN_CONTRAST:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            center = (x + width // 2, y + height // 2)
        else:
            center = (
                int(round(moments["m10"] / moments["m00"])),
                int(round(moments["m01"] / moments["m00"])),
            )

        # Favor visible, easy-to-relocate landmarks. Area helps, but cap the
        # benefit so very large faint texture does not beat clear patches.
        size_score = min(np.power(area, 0.62), 180.0)
        shape_score = 0.65 + min(circularity, 1.0)
        score = pigment_contrast * size_score * shape_score
        candidates.append(
            {
                "center": center,
                "bbox": (x, y, width, height),
                "area": area,
                "contrast": contrast,
                "color_contrast": color_contrast,
                "large_patch_contrast": large_patch_contrast,
                "pigment_contrast": pigment_contrast,
                "circularity": circularity,
                "skin_coverage": skin_coverage,
                "inner_skin_coverage": inner_skin_coverage,
                "strict_skin_coverage": strict_skin_coverage,
                "context_non_skin_fraction": context_non_skin_fraction,
                "mean_value": mean_value,
                "core_pixels": core_pixels,
                "core_fraction": core_fraction,
                "score": score,
            }
        )

    selected = []
    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        duplicate_region = False
        for existing in selected:
            distance = landmark_center_distance(candidate, existing)
            overlap = bbox_overlap_ratio(candidate, existing)
            existing_width = existing["bbox"][2]
            existing_height = existing["bbox"][3]
            candidate_width = candidate["bbox"][2]
            candidate_height = candidate["bbox"][3]
            size_based_distance = 0.6 * max(
                existing_width,
                existing_height,
                candidate_width,
                candidate_height,
                MOLE_MIN_DISTANCE,
            )

            if distance < size_based_distance or overlap > 0.15:
                duplicate_region = True
                break

        if duplicate_region:
            continue

        if nearby_selected_count(candidate, selected, radius=260) >= 2:
            continue

        selected.append(candidate)
        if len(selected) >= max_landmarks:
            break

    debug = {
        "mole_dark_response": dark_response,
        "mole_color_response": color_response,
        "mole_large_patch_response": large_patch_response,
        "mole_pigment_response": pigment_response,
        "mole_candidate_mask": dark_mask,
        "mole_strong_core_mask": strong_core_mask,
        "mole_skin_mask": skin_mask,
        "mole_strict_skin_mask": strict_skin_mask,
        "mole_skin_distance": cv2.normalize(
            skin_distance,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        ).astype(np.uint8),
        "mole_inner_skin_mask": inner_skin_mask,
        "mole_valid_region_mask": valid_region_mask,
    }
    return selected, debug


def annotate_ring(image, detection, show_raw_contours=False):
    """Draw ring outline, center dot, crosshair, and center coordinates."""
    output = image.copy()

    if detection is None:
        cv2.putText(
            output,
            "Metal ring not detected",
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
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
        return output

    ellipse = detection["ellipse"]
    (cx, cy), (_width, _height), _angle = ellipse
    center = (int(round(cx)), int(round(cy)))
    outline_thickness = ring_outline_thickness(ellipse)

    # Thick green fitted outer ellipse. This is the main ring outline.
    cv2.ellipse(output, ellipse, (0, 255, 0), outline_thickness)

    if show_raw_contours and detection.get("contour") is not None:
        cv2.drawContours(output, [detection["contour"]], -1, (0, 180, 255), 2)

    # Optional yellow inner/raw contour for debugging only.
    if show_raw_contours and detection["hole_contour"] is not None:
        cv2.drawContours(output, [detection["hole_contour"]], -1, (0, 255, 255), 2)

    # Red center dot.
    cv2.circle(output, center, 7, (0, 0, 255), -1)

    # Blue crosshair.
    cv2.line(
        output,
        (center[0] - CROSSHAIR_LENGTH, center[1]),
        (center[0] + CROSSHAIR_LENGTH, center[1]),
        (255, 0, 0),
        3,
    )
    cv2.line(
        output,
        (center[0], center[1] - CROSSHAIR_LENGTH),
        (center[0], center[1] + CROSSHAIR_LENGTH),
        (255, 0, 0),
        3,
    )

    # Red coordinate label with black outline for readability.
    label = f"Center: ({center[0]}, {center[1]})"
    text_position = (center[0] + 15, max(30, center[1] - 15))
    cv2.putText(
        output,
        label,
        text_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 0),
        5,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        label,
        text_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    return output


def annotate_mole_landmarks(image, landmarks):
    """Draw boxes around detected dark visual landmarks."""
    output = image.copy()
    image_height, image_width = output.shape[:2]

    for index, landmark in enumerate(landmarks, start=1):
        x, y, width, height = landmark["bbox"]
        color = MOLE_BOX_COLORS[(index - 1) % len(MOLE_BOX_COLORS)]

        x1 = max(0, x - MOLE_BOX_PADDING)
        y1 = max(0, y - MOLE_BOX_PADDING)
        x2 = min(image_width - 1, x + width + MOLE_BOX_PADDING)
        y2 = min(image_height - 1, y + height + MOLE_BOX_PADDING)

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)

        label = f"M{index}"
        label_position = (x1, max(20, y1 - 8))
        cv2.putText(
            output,
            label,
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            label,
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

    return output


def make_manual_landmark(x1, y1, x2, y2):
    """Create a landmark record from two dragged rectangle corners."""
    left = int(min(x1, x2))
    top = int(min(y1, y2))
    right = int(max(x1, x2))
    bottom = int(max(y1, y2))
    width = right - left
    height = bottom - top

    if width < MIN_MANUAL_LANDMARK_BOX_SIZE or height < MIN_MANUAL_LANDMARK_BOX_SIZE:
        return None

    return {
        "center": (left + width // 2, top + height // 2),
        "bbox": (left, top, width, height),
        "area": float(width * height),
        "pigment_contrast": 0.0,
        "manual": True,
    }


def manually_label_mole_landmarks(base_image, max_landmarks=MAX_MOLE_LANDMARKS):
    """
    Let the user manually draw skin-landmark boxes.

    Controls:
    - Drag left mouse button to draw a box.
    - Press u or Backspace to undo the most recent box.
    - Press c, Enter, or Space to finish.
    - Press q or Esc to cancel manual labeling.
    """
    window_name = "Manual Skin Landmarks"
    state = {
        "landmarks": [],
        "drawing": False,
        "start": None,
        "current": None,
    }

    def render():
        display = annotate_mole_landmarks(base_image, state["landmarks"])
        if state["drawing"] and state["start"] is not None and state["current"] is not None:
            color = MOLE_BOX_COLORS[len(state["landmarks"]) % len(MOLE_BOX_COLORS)]
            cv2.rectangle(display, state["start"], state["current"], color, 2)

        help_text = "Drag boxes. c/Enter/Space=finish, u=undo, q/Esc=cancel"
        cv2.putText(
            display,
            help_text,
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            help_text,
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        count_text = f"Manual landmarks: {len(state['landmarks'])}/{max_landmarks}"
        cv2.putText(
            display,
            count_text,
            (18, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            count_text,
            (18, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return display

    def mouse_callback(event, x, y, _flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(state["landmarks"]) >= max_landmarks:
                print(f"Maximum manual landmarks reached: {max_landmarks}")
                return
            state["drawing"] = True
            state["start"] = (x, y)
            state["current"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            state["current"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and state["drawing"]:
            state["drawing"] = False
            landmark = make_manual_landmark(
                state["start"][0],
                state["start"][1],
                x,
                y,
            )
            state["start"] = None
            state["current"] = None
            if landmark is not None:
                state["landmarks"].append(landmark)

    print("Manual landmark labeling opened.")
    print("Drag with the left mouse button to draw boxes.")
    print("Press u/Backspace to undo, c/Enter/Space to finish, q/Esc to cancel.")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    try:
        while True:
            cv2.imshow(window_name, render())
            key = cv2.waitKey(20) & 0xFF

            if key in (ord("c"), ord(" "), 13):
                break
            if key in (ord("u"), 8, 127):
                if state["landmarks"]:
                    state["landmarks"].pop()
            if key in (ord("q"), 27):
                raise RuntimeError("Manual landmark labeling cancelled.")
    finally:
        cv2.destroyWindow(window_name)

    return state["landmarks"]


def save_manual_landmarks(image_path, landmarks):
    """Save manually drawn landmarks as JSON next to the annotated output."""
    image_path = Path(image_path)
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{image_path.stem}_manual_landmarks.json"
    records = []

    for index, landmark in enumerate(landmarks, start=1):
        x, y, width, height = landmark["bbox"]
        center_x, center_y = landmark["center"]
        records.append(
            {
                "label": f"M{index}",
                "center": {"x": center_x, "y": center_y},
                "bbox": {"x": x, "y": y, "width": width, "height": height},
            }
        )

    with path.open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2)

    return path


def save_outputs(image_path, annotated, debug, save_images=False):
    """Save the annotated image and debug images."""
    if not save_images:
        return None, {}

    image_path = Path(image_path)
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    annotated_path = output_dir / f"{image_path.stem}_metal_ring_annotated.jpg"
    debug_paths = {}

    cv2.imwrite(str(annotated_path), annotated)

    for name, debug_image in debug.items():
        path = output_dir / f"{image_path.stem}_metal_ring_{name}.jpg"
        cv2.imwrite(str(path), debug_image)
        debug_paths[name] = path

    return annotated_path, debug_paths


def print_detection_summary(
    detection,
    mole_landmarks,
    debug_paths,
    annotated_path,
    no_moles=False,
    ring_skipped=False,
):
    """Print a concise summary of the ring and landmark detection results."""
    if ring_skipped:
        print("Metal ring detection skipped.")
    elif detection is None:
        print("No metal ring detected.")
        print("Try tuning HSV thresholds, MIN_RING_AREA, or hole-ratio settings.")
    else:
        (cx, cy), (width, height), angle = detection["ellipse"]
        print("Metal ring detected.")
        print(f"Center: ({int(round(cx))}, {int(round(cy))})")
        print(f"Ellipse width={width:.1f}, height={height:.1f}, angle={angle:.1f}")
        print(f"Detection method={detection.get('method', 'unknown')}")
        print(f"Hole ratio={detection['hole_ratio']:.2f}")
        print(f"Candidate score={detection['score']:.1f}")

    manual_mode = any(landmark.get("manual") for landmark in mole_landmarks)

    if no_moles:
        print("Dark visual landmark detection skipped.")
    elif manual_mode:
        print(f"Manual skin landmarks: {len(mole_landmarks)}")
        for index, landmark in enumerate(mole_landmarks, start=1):
            center_x, center_y = landmark["center"]
            x, y, width, height = landmark["bbox"]
            print(
                f"  M{index}: center=({center_x}, {center_y}), "
                f"bbox=({x}, {y}, {width}, {height})"
            )
    else:
        print(f"Detected dark visual landmarks: {len(mole_landmarks)}")
        for index, landmark in enumerate(mole_landmarks, start=1):
            center_x, center_y = landmark["center"]
            print(
                f"  M{index}: center=({center_x}, {center_y}), "
                f"area={landmark['area']:.1f}, "
                f"pigment_contrast={landmark['pigment_contrast']:.1f}"
            )

    if annotated_path is None:
        print("Output image saving skipped.")
    else:
        print(f"Saved annotated image: {annotated_path}")
        print("Saved debug images:")
        for name, path in debug_paths.items():
            print(f"  {name}: {path}")


def process_image(image, image_path, args):
    """Run ring/landmark detection, save outputs, and return display data."""
    if args.no_ring:
        detection = None
        debug = {}
        annotated = image.copy()
    else:
        detection, debug = detect_ring(image)
        annotated = annotate_ring(image, detection, show_raw_contours=False)

    mole_landmarks = []
    if args.manual_moles:
        mole_landmarks = manually_label_mole_landmarks(
            annotated,
            max_landmarks=max(0, args.max_moles),
        )
        if args.save_images:
            manual_path = save_manual_landmarks(image_path, mole_landmarks)
            print(f"Saved manual landmarks: {manual_path}")
        annotated = annotate_mole_landmarks(annotated, mole_landmarks)
    elif not args.no_moles:
        mole_landmarks, mole_debug = detect_mole_landmarks(
            image,
            ring_detection=detection,
            max_landmarks=max(0, args.max_moles),
        )
        debug.update(mole_debug)
        annotated = annotate_mole_landmarks(annotated, mole_landmarks)

    annotated_path, debug_paths = save_outputs(
        image_path,
        annotated,
        debug,
        save_images=args.save_images,
    )
    return detection, mole_landmarks, debug, annotated, annotated_path, debug_paths


def show_debug_windows(annotated, debug):
    """Show the annotated result and key debug masks in OpenCV windows."""
    cv2.imshow("Annotated Metal Ring", annotated)
    if "metal_mask" in debug:
        cv2.imshow("Debug Metal Mask", debug["metal_mask"])
    if "mole_skin_mask" in debug:
        cv2.imshow("Debug Mole Skin Mask", debug["mole_skin_mask"])
    if "mole_valid_region_mask" in debug:
        cv2.imshow("Debug Mole Valid Region", debug["mole_valid_region_mask"])
    print("Press any key in an OpenCV window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def image_path_to_url(path):
    """Convert a local output path to a URL served by the GUI."""
    path = Path(path)
    try:
        relative = path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        relative = path
    return "/" + urllib.parse.quote(str(relative).replace("\\", "/"))


def image_to_data_url(image, extension=".jpg"):
    """Encode an OpenCV image as a browser-displayable data URL."""
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise RuntimeError("Could not encode annotated image for preview.")
    mime_type = "image/png" if extension.lower() == ".png" else "image/jpeg"
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


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


def process_gui_image(
    image,
    image_path,
    ring_mode,
    landmark_mode,
    max_moles,
    manual_boxes,
    save_images=False,
):
    """Run the detector for the localhost GUI without opening OpenCV windows."""
    ring_skipped = ring_mode == "skip"
    if ring_skipped:
        detection = None
        debug = {}
        annotated = image.copy()
    else:
        detection, debug = detect_ring(image)
        annotated = annotate_ring(image, detection, show_raw_contours=False)

    mole_landmarks = []
    manual_landmarks_path = None
    if landmark_mode == "auto":
        mole_landmarks, mole_debug = detect_mole_landmarks(
            image,
            ring_detection=detection,
            max_landmarks=max(0, max_moles),
        )
        debug.update(mole_debug)
        annotated = annotate_mole_landmarks(annotated, mole_landmarks)
    elif landmark_mode == "manual":
        mole_landmarks = landmarks_from_gui_boxes(manual_boxes)
        if max_moles >= 0:
            mole_landmarks = mole_landmarks[:max_moles]
        if save_images:
            manual_landmarks_path = save_manual_landmarks(image_path, mole_landmarks)
        annotated = annotate_mole_landmarks(annotated, mole_landmarks)

    annotated_path, debug_paths = save_outputs(
        image_path,
        annotated,
        debug,
        save_images=save_images,
    )
    return {
        "detection": detection,
        "landmarks": mole_landmarks,
        "annotated": annotated,
        "annotated_path": annotated_path,
        "debug_paths": debug_paths,
        "manual_landmarks_path": manual_landmarks_path,
        "save_images": save_images,
        "ring_skipped": ring_skipped,
    }


def detection_to_gui_summary(result, landmark_mode):
    """Create a JSON-friendly result summary for the GUI."""
    detection = result["detection"]
    landmarks = result["landmarks"]
    ring = {"detected": False, "skipped": bool(result.get("ring_skipped"))}
    if detection is not None:
        (cx, cy), (width, height), angle = detection["ellipse"]
        ring = {
            "detected": True,
            "skipped": False,
            "center": [int(round(cx)), int(round(cy))],
            "width": round(float(width), 1),
            "height": round(float(height), 1),
            "angle": round(float(angle), 1),
            "method": detection.get("method", "unknown"),
            "score": round(float(detection["score"]), 1),
            "hole_ratio": round(float(detection["hole_ratio"]), 2),
        }

    landmark_records = []
    for index, landmark in enumerate(landmarks, start=1):
        x, y, width, height = landmark["bbox"]
        center_x, center_y = landmark["center"]
        landmark_records.append(
            {
                "label": f"M{index}",
                "center": [center_x, center_y],
                "bbox": [x, y, width, height],
                "area": round(float(landmark.get("area", width * height)), 1),
                "pigment_contrast": round(float(landmark.get("pigment_contrast", 0.0)), 1),
                "manual": bool(landmark.get("manual")),
            }
        )

    debug_urls = {
        name: image_path_to_url(path)
        for name, path in result["debug_paths"].items()
    }
    annotated_url = (
        image_path_to_url(result["annotated_path"])
        if result["annotated_path"] is not None
        else image_to_data_url(result["annotated"])
    )
    payload = {
        "ring": ring,
        "landmark_mode": landmark_mode,
        "landmarks": landmark_records,
        "annotated_url": annotated_url,
        "debug_urls": debug_urls,
        "saved_outputs": bool(result["save_images"]),
    }
    if result["manual_landmarks_path"] is not None:
        payload["manual_landmarks_url"] = image_path_to_url(result["manual_landmarks_path"])
    return payload


COMBINED_GUI_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OpenCV Clinical Tools</title>
  <style>
    * { box-sizing: border-box; }
    body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:#eef2f6; color:#1f2933; }
    header { height:56px; display:flex; align-items:center; justify-content:space-between; padding:0 18px; background:#17212b; color:white; }
    h1 { margin:0; font-size:18px; font-weight:650; letter-spacing:0; }
    nav { display:flex; gap:8px; }
    button { border:1px solid #405263; background:#243241; color:white; border-radius:6px; padding:8px 11px; cursor:pointer; font-weight:650; }
    button.active { background:#0f766e; border-color:#0f766e; }
    iframe { width:100%; height:calc(100vh - 56px); border:0; display:none; background:white; }
    iframe.active { display:block; }
  </style>
</head>
<body>
  <header>
    <h1>OpenCV Clinical Tools</h1>
    <nav>
      <button id="metalTab" class="active" type="button">Metal Ring and Landmarks</button>
      <button id="alignmentTab" type="button">Live Reference Alignment</button>
    </nav>
  </header>
  <iframe id="metalFrame" class="active" src="/metal"></iframe>
  <iframe id="alignmentFrame" data-src="/alignment"></iframe>
  <script>
    const tabs = [
      ['metalTab', 'metalFrame'],
      ['alignmentTab', 'alignmentFrame']
    ];
    for (const [buttonId, frameId] of tabs) {
      document.getElementById(buttonId).onclick = () => {
        const frame = document.getElementById(frameId);
        if (!frame.src && frame.dataset.src) frame.src = frame.dataset.src;
        for (const [b, f] of tabs) {
          document.getElementById(b).classList.toggle('active', b === buttonId);
          document.getElementById(f).classList.toggle('active', f === frameId);
        }
      };
    }
  </script>
</body>
</html>"""


GUI_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Metal Ring and Skin Landmark Labeler</title>
  <style>
    :root { color-scheme: light; --ink:#1f2933; --muted:#65758b; --line:#cfd8e3; --panel:#f7f9fb; --accent:#0f766e; --warn:#b45309; }
    * { box-sizing: border-box; }
    body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--ink); background:#eef2f6; }
    header { padding:18px 28px; background:#fff; border-bottom:1px solid var(--line); display:flex; align-items:center; justify-content:space-between; gap:18px; }
    h1 { font-size:21px; margin:0; font-weight:650; letter-spacing:0; }
    main { display:grid; grid-template-columns:360px 1fr; gap:18px; padding:18px; max-width:1440px; margin:0 auto; }
    section { background:#fff; border:1px solid var(--line); border-radius:8px; padding:16px; }
    h2 { font-size:15px; margin:0 0 12px; }
    label { display:block; font-size:13px; font-weight:600; margin:12px 0 6px; color:#314154; }
    input, select, button { font:inherit; }
    input[type=file], input[type=number], select { width:100%; border:1px solid var(--line); border-radius:6px; padding:9px; background:#fff; }
    button { border:1px solid #0f766e; background:var(--accent); color:white; border-radius:6px; padding:9px 11px; cursor:pointer; font-weight:650; }
    button.secondary { background:#fff; color:var(--accent); }
    button.warn { border-color:var(--warn); color:var(--warn); background:#fff; }
    button:disabled { opacity:.55; cursor:not-allowed; }
    .row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
    .hint { color:var(--muted); font-size:12px; line-height:1.4; }
    .status { padding:10px 12px; background:var(--panel); border:1px solid var(--line); border-radius:6px; font-size:13px; margin-top:12px; white-space:pre-wrap; }
    .workspace { display:grid; grid-template-columns:minmax(320px,1fr) minmax(320px,1fr); gap:18px; }
    .frame { position:relative; width:100%; background:#111827; border:1px solid var(--line); border-radius:8px; overflow:hidden; min-height:320px; display:flex; align-items:center; justify-content:center; }
    .frame img, .frame video, .frame canvas { max-width:100%; height:auto; display:block; }
    #manualCanvas { cursor:crosshair; background:#111827; }
    .metrics { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; margin-top:12px; }
    .metric { background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:10px; }
    .metric b { display:block; font-size:18px; margin-bottom:2px; }
    .metric span { color:var(--muted); font-size:12px; }
    table { width:100%; border-collapse:collapse; margin-top:12px; font-size:13px; }
    th, td { text-align:left; border-bottom:1px solid var(--line); padding:7px 5px; }
    th { color:#42526a; font-size:12px; }
    .debug { display:flex; gap:8px; flex-wrap:wrap; margin-top:12px; }
    .debug a { color:var(--accent); font-size:12px; }
    @media (max-width: 980px) { main, .workspace { grid-template-columns:1fr; } header { align-items:flex-start; flex-direction:column; } }
  </style>
</head>
<body>
  <header>
    <h1>Metal Ring and Skin Landmark Labeler</h1>
    <div class="hint">Local browser interface for non-diagnostic visual landmark labeling.</div>
  </header>
  <main>
    <section>
      <h2>Input</h2>
      <label>Image file</label>
      <input id="fileInput" type="file" accept="image/*">
      <label>Camera</label>
      <div class="row">
        <button type="button" class="secondary" id="startCamera">Start camera</button>
        <button type="button" class="secondary" id="captureCamera" disabled>Capture</button>
      </div>
      <video id="video" playsinline muted style="width:100%; margin-top:10px; display:none;"></video>
      <canvas id="cameraCanvas" style="display:none;"></canvas>

      <h2 style="margin-top:18px;">Ring</h2>
      <label>Ring mode</label>
      <select id="ringMode">
        <option value="detect">Detect metal ring</option>
        <option value="skip">Do not look for ring</option>
      </select>

      <h2 style="margin-top:18px;">Landmarks</h2>
      <label>Landmark mode</label>
      <select id="landmarkMode">
        <option value="auto">Automatic skin landmarks</option>
        <option value="manual">Manual skin landmarks</option>
        <option value="none">Ring only</option>
      </select>
      <label>Maximum landmarks</label>
      <input id="maxMoles" type="number" min="0" max="30" value="7">
      <label><input id="saveImages" type="checkbox"> Save output images</label>
      <div class="row" style="margin-top:14px;">
        <button type="button" id="runButton">Run labeling</button>
        <button type="button" class="warn" id="undoBox">Undo box</button>
        <button type="button" class="secondary" id="clearBoxes">Clear boxes</button>
      </div>
      <div class="status" id="status">Choose an image or capture from the camera.</div>
    </section>

    <section>
      <div class="workspace">
        <div>
          <h2>Source and Manual Boxes</h2>
          <div class="frame"><canvas id="manualCanvas"></canvas></div>
          <div class="hint" style="margin-top:8px;">Manual mode: drag on the image to create M1, M2, M3... boxes before running.</div>
        </div>
        <div>
          <h2>Annotated Output</h2>
          <div class="frame"><img id="resultImage" alt=""></div>
          <div class="metrics">
            <div class="metric"><b id="ringState">-</b><span>Ring</span></div>
            <div class="metric"><b id="landmarkCount">-</b><span>Landmarks</span></div>
            <div class="metric"><b id="ringCenter">-</b><span>Ring center</span></div>
          </div>
          <table id="landmarkTable"></table>
          <div class="debug" id="debugLinks"></div>
        </div>
      </div>
    </section>
  </main>
<script>
const fileInput = document.getElementById('fileInput');
const manualCanvas = document.getElementById('manualCanvas');
const ctx = manualCanvas.getContext('2d');
const statusBox = document.getElementById('status');
const resultImage = document.getElementById('resultImage');
const video = document.getElementById('video');
const cameraCanvas = document.getElementById('cameraCanvas');
let sourceFile = null, sourceDataUrl = '', imageBitmap = null, boxes = [], drawing = null, stream = null;

function setStatus(text) { statusBox.textContent = text; }
function drawManualCanvas() {
  const maxW = 980;
  if (!imageBitmap) {
    manualCanvas.width = 640; manualCanvas.height = 420;
    ctx.fillStyle = '#111827'; ctx.fillRect(0,0,manualCanvas.width,manualCanvas.height);
    ctx.fillStyle = '#d7dee8'; ctx.font = '16px sans-serif'; ctx.fillText('No image loaded', 24, 42);
    return;
  }
  const scale = Math.min(1, maxW / imageBitmap.width);
  manualCanvas.width = Math.round(imageBitmap.width * scale);
  manualCanvas.height = Math.round(imageBitmap.height * scale);
  ctx.drawImage(imageBitmap, 0, 0, manualCanvas.width, manualCanvas.height);
  [...boxes, drawing].filter(Boolean).forEach((b, i) => {
    ctx.strokeStyle = ['#ff00ff','#00ffff','#ffa500','#ffff00','#00ff00','#0000ff','#ffffff'][i % 7];
    ctx.lineWidth = 3; ctx.strokeRect(b.x*scale, b.y*scale, b.width*scale, b.height*scale);
    ctx.font = '18px sans-serif'; ctx.lineWidth = 4; ctx.strokeText('M'+(i+1), b.x*scale, Math.max(20, b.y*scale-6));
    ctx.fillStyle = ctx.strokeStyle; ctx.fillText('M'+(i+1), b.x*scale, Math.max(20, b.y*scale-6));
  });
}
function canvasEventToImagePoint(e) {
  const r = manualCanvas.getBoundingClientRect();
  const x = (e.clientX - r.left) * imageBitmap.width / r.width;
  const y = (e.clientY - r.top) * imageBitmap.height / r.height;
  return {
    x: Math.max(0, Math.min(imageBitmap.width, x)),
    y: Math.max(0, Math.min(imageBitmap.height, y))
  };
}
async function loadDataUrl(dataUrl, file=null) {
  sourceDataUrl = dataUrl; sourceFile = file; boxes = [];
  imageBitmap = await createImageBitmap(await (await fetch(dataUrl)).blob());
  drawManualCanvas();
  setStatus('Image loaded. Choose landmark mode and run labeling.');
}
fileInput.addEventListener('change', () => {
  const file = fileInput.files[0]; if (!file) return;
  const reader = new FileReader();
  reader.onload = () => loadDataUrl(reader.result, file);
  reader.readAsDataURL(file);
});
manualCanvas.addEventListener('mousedown', e => {
  if (!imageBitmap || document.getElementById('landmarkMode').value !== 'manual') return;
  const p = canvasEventToImagePoint(e);
  drawing = {x:p.x, y:p.y, width:0, height:0, startX:p.x, startY:p.y};
});
manualCanvas.addEventListener('mousemove', e => {
  if (!drawing) return;
  const p = canvasEventToImagePoint(e);
  drawing.x = Math.min(drawing.startX, p.x); drawing.y = Math.min(drawing.startY, p.y);
  drawing.width = Math.abs(p.x - drawing.startX); drawing.height = Math.abs(p.y - drawing.startY);
  drawManualCanvas();
});
window.addEventListener('mouseup', () => {
  if (!drawing) return;
  if (drawing.width >= 4 && drawing.height >= 4) boxes.push({x:Math.round(drawing.x), y:Math.round(drawing.y), width:Math.round(drawing.width), height:Math.round(drawing.height)});
  drawing = null; drawManualCanvas();
});
document.getElementById('undoBox').onclick = () => { boxes.pop(); drawManualCanvas(); };
document.getElementById('clearBoxes').onclick = () => { boxes = []; drawManualCanvas(); };
document.getElementById('startCamera').onclick = async () => {
  stream = await navigator.mediaDevices.getUserMedia({video:true});
  video.srcObject = stream; video.style.display = 'block'; await video.play();
  document.getElementById('captureCamera').disabled = false;
};
document.getElementById('captureCamera').onclick = async () => {
  cameraCanvas.width = video.videoWidth; cameraCanvas.height = video.videoHeight;
  cameraCanvas.getContext('2d').drawImage(video,0,0);
  await loadDataUrl(cameraCanvas.toDataURL('image/jpeg', .94), null);
};
document.getElementById('runButton').onclick = async () => {
  if (!sourceDataUrl) { setStatus('Load or capture an image first.'); return; }
  const form = new FormData();
  form.append('ring_mode', document.getElementById('ringMode').value);
  form.append('landmark_mode', document.getElementById('landmarkMode').value);
  form.append('max_moles', document.getElementById('maxMoles').value);
  form.append('save_images', document.getElementById('saveImages').checked ? '1' : '0');
  form.append('manual_boxes', JSON.stringify(boxes));
  if (sourceFile) form.append('image', sourceFile); else form.append('image_data', sourceDataUrl);
  setStatus('Running detector...');
  const res = await fetch('/api/analyze', {method:'POST', body:form});
  const data = await res.json();
  if (!res.ok) { setStatus(data.error || 'Analysis failed.'); return; }
  resultImage.src = data.annotated_url.startsWith('data:') ? data.annotated_url : data.annotated_url + '?t=' + Date.now();
  document.getElementById('ringState').textContent = data.ring.skipped ? 'Skipped' : (data.ring.detected ? 'Detected' : 'Not found');
  document.getElementById('landmarkCount').textContent = data.landmarks.length;
  document.getElementById('ringCenter').textContent = data.ring.detected ? data.ring.center.join(', ') : '-';
  document.getElementById('landmarkTable').innerHTML = '<tr><th>Label</th><th>Center</th><th>Box</th></tr>' + data.landmarks.map(l => `<tr><td>${l.label}</td><td>${l.center.join(', ')}</td><td>${l.bbox.join(', ')}</td></tr>`).join('');
  document.getElementById('debugLinks').innerHTML = Object.entries(data.debug_urls).map(([k,v]) => `<a href="${v}" target="_blank">${k}</a>`).join('');
  setStatus(data.saved_outputs ? 'Done. Annotated image and debug outputs saved in output/.' : 'Done. Preview generated without saving output images.');
};
drawManualCanvas();
</script>
</body>
</html>"""


class MetalRingGuiHandler(SimpleHTTPRequestHandler):
    """HTTP handler for the built-in localhost GUI."""

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self.send_html(COMBINED_GUI_PAGE)
            return
        if parsed.path == "/metal":
            self.send_html(GUI_PAGE)
            return
        if parsed.path == "/alignment":
            self.send_html(alignment_gui.WEB_APP_HTML)
            return
        if parsed.path == "/stream":
            self.send_alignment_stream()
            return
        if parsed.path == "/api/state":
            self.send_json(self.server.state.state_payload())
            return
        if parsed.path.startswith("/output/"):
            return super().do_GET()
        self.send_error(404, "Not found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/analyze":
            try:
                payload = self.read_analysis_request()
                result = process_gui_image(**payload)
                response = detection_to_gui_summary(result, payload["landmark_mode"])
                self.send_json(response)
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=500)
            return

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
                self.handle_alignment_reference_upload()
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

    def send_html(self, html):
        data = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        body = self.rfile.read(length).decode("utf-8")
        return json.loads(body)

    def handle_alignment_reference_upload(self):
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
        data = urllib.parse.parse_qs(self.rfile.read(length).decode("utf-8"))
        path = data.get("path", [""])[0]
        if not path:
            self.send_json({"ok": False, "message": "No reference path was provided."}, status=400)
            return
        self.server.state.load_reference_path(path)
        self.send_json({"ok": True, "message": f"Loaded reference image: {path}"})

    def send_alignment_stream(self):
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

    def read_analysis_request(self):
        form = FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
            },
        )
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        save_images = form.getfirst("save_images", "0") == "1"
        upload_dir = Path("output") / "gui_uploads"
        if save_images:
            upload_dir.mkdir(parents=True, exist_ok=True)

        image_field = form["image"] if "image" in form else None
        image_data = form.getfirst("image_data", "")
        if image_field is not None and getattr(image_field, "filename", ""):
            suffix = Path(image_field.filename).suffix or ".jpg"
            image_path = upload_dir / f"{run_id}{suffix}"
            image_bytes = image_field.file.read()
            if save_images:
                image_path.write_bytes(image_bytes)
                image = load_image(image_path)
            else:
                image_path = Path(f"gui_upload_{run_id}{suffix}")
                image_array = np.frombuffer(image_bytes, dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError("Could not decode uploaded image.")
        elif image_data:
            header, encoded = image_data.split(",", 1) if "," in image_data else ("", image_data)
            image_bytes = base64.b64decode(encoded)
            image_path = upload_dir / f"{run_id}_camera.jpg"
            if save_images:
                image_path.write_bytes(image_bytes)
            else:
                image_path = Path(f"gui_camera_{run_id}.jpg")
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError("Could not decode camera image.")
        else:
            raise RuntimeError("No image was provided.")

        try:
            max_moles = int(form.getfirst("max_moles", str(MAX_MOLE_LANDMARKS)))
        except ValueError:
            max_moles = MAX_MOLE_LANDMARKS
        landmark_mode = form.getfirst("landmark_mode", "auto")
        if landmark_mode not in {"auto", "manual", "none"}:
            landmark_mode = "auto"
        ring_mode = form.getfirst("ring_mode", "detect")
        if ring_mode not in {"detect", "skip"}:
            ring_mode = "detect"
        manual_boxes = json.loads(form.getfirst("manual_boxes", "[]"))
        if not isinstance(manual_boxes, list):
            manual_boxes = []

        return {
            "image": image,
            "image_path": image_path,
            "ring_mode": ring_mode,
            "landmark_mode": landmark_mode,
            "max_moles": max(0, max_moles),
            "manual_boxes": manual_boxes,
            "save_images": save_images,
        }

    def send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def create_alignment_args(host, port, reference_image=None, camera_index=0):
    """Create default arguments for the embedded live reference alignment GUI."""
    return SimpleNamespace(
        reference_image=reference_image,
        host=host,
        port=port,
        opencv_window=False,
        camera=camera_index,
        mirror=False,
        preview_width=960,
        capture_width=640,
        capture_height=480,
        fps=30,
        fit_mode="cover",
        orb_features=1200,
        max_matches=alignment_gui.DEFAULT_MAX_MATCHES,
        match_mode="orb",
        max_markers=alignment_gui.DEFAULT_MAX_MARKERS,
        marker_detect_scale=alignment_gui.DEFAULT_MARKER_DETECT_SCALE,
        visual_score_scale=alignment_gui.DEFAULT_VISUAL_SCORE_SCALE,
        score_every=8,
        select_roi=False,
        draw_mask=False,
        skin_mask=False,
        no_auto_target_mask=False,
        output_dir="output",
    )


class CombinedGuiServer(ThreadingHTTPServer):
    """Server with shared live-alignment state plus metal-ring routes."""

    def __init__(self, address, handler_class, alignment_state):
        super().__init__(address, handler_class)
        self.state = alignment_state


def run_gui_server(
    host="127.0.0.1",
    port=8502,
    alignment_reference=None,
    camera_index=0,
    open_browser=False,
):
    """Start the built-in local browser GUI."""
    alignment_args = create_alignment_args(
        host,
        port,
        reference_image=alignment_reference,
        camera_index=camera_index,
    )
    alignment_state = alignment_gui.AlignmentWebState(alignment_args)
    try:
        server = CombinedGuiServer((host, port), MetalRingGuiHandler, alignment_state)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            url = f"http://{host}:{port}"
            raise SystemExit(
                f"Port {port} is already in use.\n"
                f"If the GUI is already open, use {url}.\n"
                "Otherwise stop the earlier server with Control+C in its terminal, "
                f"or start a second copy with: python3 detect_metal_ring_photo.py --gui --port {port + 1}"
            ) from exc
        raise

    url = f"http://{host}:{port}"
    print(f"Metal ring GUI running at {url}")
    print("Press Control+C in this terminal to stop the server.")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping GUI server.")
    finally:
        alignment_state.stop_camera()
        server.server_close()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Detect a metallic ring and dark visual skin landmarks in a still photo."
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        help="Path to the image containing the metal ring. Omit when using --webcam.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Start the built-in localhost browser GUI.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for --gui. Default: 127.0.0.1.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8502,
        help="Port for --gui. Default: 8502.",
    )
    parser.add_argument(
        "--alignment-reference",
        help="Optional reference image path to preload in the live reference alignment tab.",
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Open a webcam preview, capture one photo, then run detection on it.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Webcam device index to use with --webcam. Default: 0.",
    )
    parser.add_argument(
        "--capture-path",
        help="Optional path for the raw webcam capture. Default: output/webcam_capture_TIMESTAMP.jpg.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the annotated image and debug mask in OpenCV windows.",
    )
    parser.add_argument(
        "--max-moles",
        type=int,
        default=MAX_MOLE_LANDMARKS,
        help=f"Maximum number of dark visual landmarks to box. Default: {MAX_MOLE_LANDMARKS}.",
    )
    parser.add_argument(
        "--no-moles",
        action="store_true",
        help="Only annotate the ring; do not detect dark visual landmarks.",
    )
    parser.add_argument(
        "--no-ring",
        action="store_true",
        help="Skip metal ring detection and only process skin landmarks.",
    )
    parser.add_argument(
        "--manual-moles",
        action="store_true",
        help="Manually draw skin-landmark boxes after optional ring detection.",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save annotated and debug output images. Default: preview/print results without saving.",
    )
    parser.add_argument(
        "--no-save-images",
        action="store_false",
        dest="save_images",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    running_as_packaged_app = getattr(sys, "frozen", False)

    if running_as_packaged_app and len(sys.argv) == 1:
        args.gui = True

    if args.gui:
        run_gui_server(
            host=args.host,
            port=args.port,
            alignment_reference=args.alignment_reference,
            camera_index=args.camera_index,
            open_browser=running_as_packaged_app,
        )
        return

    if args.manual_moles and args.no_moles:
        raise SystemExit("Use either --manual-moles or --no-moles, not both.")

    if args.webcam:
        image, image_path = capture_webcam_photo(
            camera_index=args.camera_index,
            capture_path=args.capture_path,
        )
    else:
        if args.image_path is None:
            raise SystemExit("Provide an image_path or use --webcam.")
        image_path = Path(args.image_path)
        image = load_image(image_path)

    detection, mole_landmarks, debug, annotated, annotated_path, debug_paths = process_image(
        image,
        image_path,
        args,
    )
    print_detection_summary(
        detection,
        mole_landmarks,
        debug_paths,
        annotated_path,
        no_moles=args.no_moles,
        ring_skipped=args.no_ring,
    )

    if args.show:
        show_debug_windows(annotated, debug)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        if getattr(sys, "frozen", False) and sys.platform.startswith("win"):
            input("The app hit an error. Press Enter to close this window...")
        raise
