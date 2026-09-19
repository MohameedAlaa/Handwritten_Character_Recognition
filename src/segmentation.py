import cv2
import numpy as np

def remove_ruling_lines(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect and remove long horizontal notebook ruling lines and vertical margin lines.

    Returns:
        cleaned (np.ndarray): Binary image with ruling lines removed and strokes repaired.
        lines_mask (np.ndarray): Mask of detected lines (255 where lines were found).
    """
    h, w = binary.shape[:2]

    # Horizontal ruling lines span across characters
    h_line_min_width = max(30, int(w * 0.15))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_line_min_width, 1))
    detected_h = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    # Vertical margin lines span across large vertical portions
    v_line_min_height = max(40, int(h * 0.35))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_line_min_height))
    detected_v = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    lines_mask = cv2.bitwise_or(detected_h, detected_v)

    if np.max(lines_mask) > 0:
        cleaned = cv2.subtract(binary, lines_mask)
        # Repair small gaps in strokes that intersected lines
        repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, repair_kernel)
    else:
        cleaned = binary.copy()

    return cleaned, lines_mask

def preprocess_for_segmentation(image: np.ndarray, return_lines_mask: bool = False):
    """
    Preprocess the image for contour detection.
    Supports both dark-on-light and light-on-dark images.
    Removes long horizontal and vertical ruling lines.

    Returns:
        binary: Cleaned binary image for contour detection (white foreground on black background).
        lines_mask (optional): Binary mask of removed ruling lines.
    """
    # Convert to grayscale if it's a color image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Invert if the background is lighter than the text
    # Assuming text occupies less than 50% of the image, the median/mean represents the background.
    if np.mean(gray) > 127:
        gray = 255 - gray

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Otsu's thresholding
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Remove ruling lines and background line artifacts
    clean_binary, lines_mask = remove_ruling_lines(binary)

    if return_lines_mask:
        return clean_binary, lines_mask
    return clean_binary

def pad_and_square_crop(image: np.ndarray, x: int, y: int, w: int, h: int, padding: int = 4) -> np.ndarray:
    """
    Crop the bounding box from the image, make it square by padding, and add a uniform margin.
    """
    # Crop the region
    crop = image[y:y+h, x:x+w]

    # Calculate padding to make it square
    diff = abs(w - h)
    top_pad, bottom_pad, left_pad, right_pad = 0, 0, 0, 0

    if w > h:
        top_pad = diff // 2
        bottom_pad = diff - top_pad
    elif h > w:
        left_pad = diff // 2
        right_pad = diff - left_pad

    # Add square padding + uniform margin
    crop_padded = cv2.copyMakeBorder(
        crop,
        top_pad + padding,
        bottom_pad + padding,
        left_pad + padding,
        right_pad + padding,
        cv2.BORDER_CONSTANT,
        value=0
    )

    return crop_padded

def segment_characters(image: np.ndarray, min_area: int = 20):
    """
    Detect and extract individual character regions from left to right.
    Robust against notebook ruling lines, margins, and background artifacts.

    Returns:
        crops: List of padded, square character image crops.
        boxes: List of bounding boxes (x, y, w, h).
        vis_image: Color image with bounding boxes drawn for visualization.
    """
    # Handle empty/blank images
    if image is None or image.size == 0 or np.max(image) == 0:
        return [], [], image

    clean_binary, lines_mask = preprocess_for_segmentation(image, return_lines_mask=True)

    # Handle fully blank binary images
    if np.max(clean_binary) == 0:
        return [], [], image

    # 1. Robust Text-Region (ROI) Detection
    # Use morphological closing and dilation to connect nearby character strokes into word blobs
    h_img, w_img = clean_binary.shape[:2]
    roi_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, int(w_img * 0.05)), max(10, int(h_img * 0.03))))
    dilated_for_roi = cv2.morphologyEx(clean_binary, cv2.MORPH_CLOSE, roi_kernel)
    dilated_for_roi = cv2.dilate(dilated_for_roi, roi_kernel, iterations=2)

    roi_contours, _ = cv2.findContours(dilated_for_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_mask = np.zeros_like(clean_binary)
    has_valid_roi = False

    for c in roi_contours:
        x, y, w, h = cv2.boundingRect(c)

        # Reject whole-page outer borders, large shadows, or scanner background
        if w > 0.85 * w_img and h > 0.85 * h_img:
            continue

        # Reject long edges (torn paper edges, borders along one side)
        if w > 0.95 * w_img or h > 0.95 * h_img:
            continue

        # Check actual foreground pixels in this ROI
        c_mask = np.zeros_like(clean_binary)
        cv2.drawContours(c_mask, [c], -1, 255, -1)
        fg_pixels = cv2.countNonZero(cv2.bitwise_and(clean_binary, c_mask))

        # Reject regions with very few actual character pixels (small noise)
        if fg_pixels < 150:
            continue

        cv2.drawContours(roi_mask, [c], -1, 255, -1)
        has_valid_roi = True

    if has_valid_roi:
        # Apply the ROI mask to isolate the writing region
        clean_binary = cv2.bitwise_and(clean_binary, roi_mask)

    # 2. Individual Character Segmentation
    # Find contours on ROI-masked binary
    contours, _ = cv2.findContours(clean_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by minimum area to remove small noise
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    if not valid_contours:
        return [], [], image

    # Get bounding boxes
    raw_boxes = [cv2.boundingRect(c) for c in valid_contours]

    # Filter out remaining line and border artifacts
    h_img, w_img = clean_binary.shape[:2]
    boxes = []
    for (x, y, w, h) in raw_boxes:
        # Reject long thin horizontal lines (ruling lines)
        if (w / max(h, 1)) > 3.5 and h < 15:
            continue
        # Reject long thin vertical lines (margin lines)
        if (h / max(w, 1)) > 6.0 and w < 10:
            continue
        # Reject whole-page outer borders
        if w > 0.85 * w_img and h > 0.85 * h_img:
            continue
        # Reject wide horizontal lines spanning large portion of image
        if w > 0.40 * w_img and (w / max(h, 1)) > 2.5:
            continue
        boxes.append((x, y, w, h))

    if not boxes:
        return [], [], image

    def group_character_components(box_list):
        if not box_list:
            return []

        def merge_boxes(b1, b2):
            x1, y1, w1, h1 = b1
            x2, y2, w2, h2 = b2
            x = min(x1, x2)
            y = min(y1, y2)
            r = max(x1 + w1, x2 + w2)
            b = max(y1 + h1, y2 + h2)
            return (x, y, r - x, b - y)

        box_list = sorted(box_list, key=lambda b: b[0])

        merged = True
        while merged:
            merged = False
            new_boxes = []
            skip_idx = set()

            for i in range(len(box_list)):
                if i in skip_idx:
                    continue

                b1 = box_list[i]
                x1, y1, w1, h1 = b1

                merged_this_round = False
                for j in range(i + 1, len(box_list)):
                    if j in skip_idx:
                        continue

                    b2 = box_list[j]
                    x2, y2, w2, h2 = b2

                    h_overlap = min(x1 + w1, x2 + w2) - max(x1, x2)
                    h_gap = max(x1, x2) - min(x1 + w1, x2 + w2)

                    v_overlap = min(y1 + h1, y2 + h2) - max(y1, y2)
                    v_gap = max(y1, y2) - min(y1 + h1, y2 + h2)

                    max_h = max(h1, h2)
                    min_h = min(h1, h2)

                    should_merge = False

                    # Case 1: Dot above/below or inside ('!', 'i', 'j', '?')
                    if h_overlap > -min(w1, w2) * 1.5:
                        if v_gap < max_h * 0.8:
                            if h_overlap > 0 or (w1 * h1 < w2 * h2 * 0.3 or w2 * h2 < w1 * h1 * 0.3):
                                merged_w = max(x1 + w1, x2 + w2) - min(x1, x2)
                                merged_h = max(y1 + h1, y2 + h2) - min(y1, y2)
                                if merged_w < merged_h * 1.5:
                                    should_merge = True

                    # Case 2: Broken characters ('H', 'K', 'W', 'M')
                    if not should_merge and h_gap >= 0 and h_gap < max_h * 0.25:
                        if v_overlap > min_h * 0.6:
                            merged_w = max(x1 + w1, x2 + w2) - min(x1, x2)
                            merged_h = max(y1 + h1, y2 + h2) - min(y1, y2)
                            aspect = merged_w / max(merged_h, 1)

                            if aspect < 1.5:
                                # Avoid merging thin adjacent characters like "ll"
                                is_thin1 = w1 < h1 * 0.3
                                is_thin2 = w2 < h2 * 0.3
                                if is_thin1 and is_thin2 and h_gap > max_h * 0.08:
                                    should_merge = False
                                else:
                                    should_merge = True

                    if should_merge:
                        new_boxes.append(merge_boxes(b1, b2))
                        skip_idx.add(j)
                        merged = True
                        merged_this_round = True
                        break

                if not merged_this_round:
                    new_boxes.append(b1)

            box_list = sorted(new_boxes, key=lambda b: b[0])

        return box_list

    # Group components belonging to the same character
    boxes = group_character_components(boxes)

    # Sort boxes from left to right (based on x-coordinate)
    boxes = sorted(boxes, key=lambda b: b[0])


    # Create crops and visualization
    crops = []

    # Use grayscale inverted image for the crop source so crops are white-on-black
    if len(image.shape) == 3:
        source_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        source_gray = image.copy()

    if np.mean(source_gray) > 127:
        source_gray = 255 - source_gray

    # If ruling lines were detected, suppress them in source_gray so crops are clean
    if np.max(lines_mask) > 0:
        source_gray[lines_mask > 0] = 0
        repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        source_gray = cv2.morphologyEx(source_gray, cv2.MORPH_CLOSE, repair_kernel)

    vis_image = cv2.cvtColor(source_gray, cv2.COLOR_GRAY2BGR)

    for (x, y, w, h) in boxes:
        # Create padded crop
        crop = pad_and_square_crop(source_gray, x, y, w, h, padding=10)
        crops.append(crop)

        # Draw on visualization image
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return crops, boxes, vis_image
