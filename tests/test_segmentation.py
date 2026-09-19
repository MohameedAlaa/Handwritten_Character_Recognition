import numpy as np
import cv2
import pytest
from src.segmentation import segment_characters

def test_empty_image():
    # Test completely empty/black image
    image = np.zeros((100, 100), dtype=np.uint8)
    crops, boxes, vis = segment_characters(image)
    assert len(crops) == 0
    assert len(boxes) == 0

def test_single_character():
    # Create an image with a single white square in the center
    image = np.zeros((100, 100), dtype=np.uint8)
    image[40:60, 40:60] = 255

    crops, boxes, vis = segment_characters(image, min_area=10)

    assert len(crops) == 1
    assert len(boxes) == 1

    # Check bounding box
    x, y, w, h = boxes[0]
    assert x == 40 and y == 40 and w == 20 and h == 20

    # Check crop is square (before resize to 28x28)
    h_c, w_c = crops[0].shape
    assert h_c == w_c

def test_multiple_characters():
    # Create an image with 3 distinct white squares, NOT in left-to-right creation order
    image = np.zeros((100, 300), dtype=np.uint8)

    # Rightmost character (drawn first)
    image[40:60, 240:260] = 255
    # Center character
    image[40:60, 140:160] = 255
    # Leftmost character
    image[40:60, 40:60] = 255

    crops, boxes, vis = segment_characters(image, min_area=10)

    assert len(crops) == 3
    assert len(boxes) == 3

    # Check left-to-right sorting
    assert boxes[0][0] == 40
    assert boxes[1][0] == 140
    assert boxes[2][0] == 240

def test_background_inversion():
    # Test dark text on white background
    image = np.ones((100, 100), dtype=np.uint8) * 255
    image[40:60, 40:60] = 0 # Black character

    crops, boxes, vis = segment_characters(image, min_area=10)

    assert len(crops) == 1
    # Check if crop has white text on black background
    assert np.mean(crops[0]) > 0
    assert crops[0].max() == 255

def test_output_types():
    image = np.zeros((100, 100), dtype=np.uint8)
    image[40:60, 40:60] = 255

    crops, boxes, vis = segment_characters(image, min_area=10)

    assert isinstance(crops, list)
    assert isinstance(boxes, list)
    assert isinstance(vis, np.ndarray)

    if len(crops) > 0:
        assert isinstance(crops[0], np.ndarray)
        assert isinstance(boxes[0], tuple)

def test_empty_ruled_notebook_lines():
    # An image containing only ruled lines and vertical margin line without characters
    image = np.ones((200, 400), dtype=np.uint8) * 240
    image[40:42, :] = 120
    image[90:92, :] = 120
    image[140:142, :] = 120
    image[:, 60:62] = 130  # vertical margin

    crops, boxes, vis = segment_characters(image, min_area=15)
    assert len(boxes) == 0, f"Expected 0 boxes from blank ruled page, got {len(boxes)}"
    assert len(crops) == 0

def test_ruled_paper_with_characters():
    # Synthetic page with ruled lines and separated handwritten characters
    image = np.ones((160, 450), dtype=np.uint8) * 255
    # Ruled notebook lines across the page
    image[45:47, :] = 150
    image[120:122, :] = 150

    # 4 distinct separated characters
    chars = ['H', 'E', 'L', 'O']
    xs = [60, 160, 260, 360]
    for c, x in zip(chars, xs):
        cv2.putText(image, c, (x, 115), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 3)

    crops, boxes, vis = segment_characters(image, min_area=20)

    # Background lines must not be returned as character boxes
    assert len(boxes) == 4, f"Expected exactly 4 character boxes, got {len(boxes)}"
    assert len(crops) == 4

    # Verify left-to-right order
    for i in range(len(boxes) - 1):
        assert boxes[i][0] < boxes[i+1][0], "Boxes must be sorted left-to-right"

    # Verify none of the boxes have extreme aspect ratios typical of lines
    for (x, y, w, h) in boxes:
        assert (w / max(h, 1)) <= 3.5, f"Box ({x},{y},{w},{h}) resembles a horizontal line"

def test_intersecting_ruling_line():
    # A ruling line that passes directly across characters
    image = np.ones((150, 350), dtype=np.uint8) * 255
    # Ruling line intersecting the characters
    image[75:77, :] = 140

    # Two characters
    cv2.putText(image, 'A', (80, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 3)
    cv2.putText(image, 'B', (200, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 3)

    crops, boxes, vis = segment_characters(image, min_area=20)
    assert len(boxes) == 2, f"Expected 2 character boxes, got {len(boxes)}"
    assert boxes[0][0] < boxes[1][0]

def test_real_page_artifacts():
    # Synthetic full-page with borders, shadows, background artifacts, ruling lines, and one handwritten word
    image = np.ones((800, 600), dtype=np.uint8) * 240

    # 1. Page borders (left and right dark borders)
    image[:, :20] = 50
    image[:, -20:] = 50
    # Top border
    image[:20, :] = 50

    # 2. Large shadow on the bottom right
    cv2.circle(image, (500, 700), 150, 100, -1)

    # 3. Ruling lines
    for y in range(100, 800, 50):
        image[y:y+2, :] = 150

    # 4. Background artifacts (small noise)
    image[400:410, 100:110] = 80
    image[200:205, 500:505] = 90

    # 5. One handwritten word ("HELLO")
    chars = ['H', 'E', 'L', 'L', 'O']
    xs = [100, 180, 260, 340, 420]
    for c, x in zip(chars, xs):
        cv2.putText(image, c, (x, 300), cv2.FONT_HERSHEY_SIMPLEX, 2.5, 0, 5)

    crops, boxes, vis = segment_characters(image, min_area=30)

    # Verify exactly 5 character boxes are detected, ignoring all artifacts
    assert len(boxes) == 5, f"Expected 5 character boxes, got {len(boxes)}"
    assert len(crops) == 5

    # Verify left-to-right order
    for i in range(len(boxes) - 1):
        assert boxes[i][0] < boxes[i+1][0]

def test_component_grouping():
    # Synthetic "H E L L O !" image (height 400 so vertical strokes are not treated as full-page margin lines)
    image = np.ones((400, 600), dtype=np.uint8) * 255

    # "H" split into three components: left bar, right bar, and a tiny horizontal gap
    # Left bar
    cv2.rectangle(image, (50, 50), (60, 150), 0, -1)
    # Right bar
    cv2.rectangle(image, (90, 50), (100, 150), 0, -1)
    # Horizontal bar connecting them (but slightly broken)
    cv2.rectangle(image, (60, 95), (85, 105), 0, -1)

    # "E"
    cv2.putText(image, 'E', (150, 130), cv2.FONT_HERSHEY_SIMPLEX, 3.0, 0, 8)

    # "L" and "L" (must not be merged together!)
    cv2.putText(image, 'L', (250, 130), cv2.FONT_HERSHEY_SIMPLEX, 3.0, 0, 8)
    cv2.putText(image, 'L', (330, 130), cv2.FONT_HERSHEY_SIMPLEX, 3.0, 0, 8)

    # "O"
    cv2.putText(image, 'O', (420, 130), cv2.FONT_HERSHEY_SIMPLEX, 3.0, 0, 8)

    # "!" (line and a dot)
    cv2.rectangle(image, (520, 50), (530, 120), 0, -1)
    cv2.circle(image, (525, 140), 6, 0, -1)

    crops, boxes, vis = segment_characters(image, min_area=10)

    # Expected: H, E, L, L, O, ! -> 6 grouped characters
    assert len(boxes) == 6, f"Expected 6 characters, got {len(boxes)}"
    assert len(crops) == 6

    # Verify H is merged properly (x roughly 50, w roughly 50)
    assert abs(boxes[0][0] - 50) < 5
    assert abs(boxes[0][2] - 50) < 5

    # Verify "!" is merged (dot and line combined)
    # y should start around 50, and bottom (y+h) around 146
    x_excl, y_excl, w_excl, h_excl = boxes[5]
    assert abs(y_excl - 50) < 5
    assert abs(y_excl + h_excl - 146) < 10

    # Verify L and L remain separate
    # H(0), E(1), L(2), L(3)
    assert boxes[2][0] < boxes[3][0]

    # Check left-to-right order
    for i in range(len(boxes) - 1):
        assert boxes[i][0] < boxes[i+1][0]
