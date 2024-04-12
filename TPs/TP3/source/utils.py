import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def _have_same_shape(list_imgs):
    """
    Args:
        list_imgs (list of numpy.ndarray)

    Returns:
        None
    """
    for i in range(len(list_imgs)):
        next_idx = i + 1 if i < (len(list_imgs) - 1) else 0
        assert list_imgs[i].shape == list_imgs[next_idx].shape


def load_imgs(folder_path, is_gray=False):
    """
    Args:
        folder_path (str)
        flag (int)

    Returns:
        list
    """
    flag = 0 if is_gray else 1
    list_imgs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            file_path = os.path.join(folder_path, filename)

            img = cv2.imread(file_path, flag)
            list_imgs.append(img)

    _have_same_shape(list_imgs)

    return list_imgs


def plot_imgs(
    list_imgs,
    list_imgs_title=None,
    title="",
    rows=1,
    cols=None,
    figsize=5,
    is_gray=False,
    cmap="gray",
):
    """
    Args:
        list_imgs (list)
        title (str)
        rows (int)
        cols(int)

    Returns:
        None
    """
    cols = len(list_imgs) if not cols else cols

    fig, axes = plt.subplots(rows, cols, figsize=(figsize * cols, figsize * rows))

    fig.suptitle(title, fontsize=16, fontweight="bold")

    for i, image in enumerate(list_imgs):
        if is_gray:
            axes[i].imshow(image, cmap=cmap)
        else:
            axes[i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if list_imgs_title:
            axes[i].set_title(list_imgs_title[i], fontsize=int(figsize * 2.5))
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def box_iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area_a = box_area(boxes_a.T)
    area_b = box_area(boxes_b.T)

    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[:, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[:, 2:])

    area_inter = np.prod(np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)

    return area_inter / (area_a[:, None] + area_b - area_inter)


def non_max_suppression(matches, iou_threshold=0.8) -> np.ndarray:

    def is_contained(box_a, box_b):
        x1_in = box_a[0] > box_b[0]
        y1_in = box_a[1] > box_b[1]
        x2_in = box_a[2] < box_b[2]
        y2_in = box_a[3] < box_b[3]

        contained_anchors = [x1_in, y1_in, x2_in, y2_in]

        is_contained = len(set(contained_anchors)) == 1

        return is_contained

    rows = matches.shape[0]

    sort_index = np.flip(matches[:, 4].argsort())
    matches_sorted = matches[sort_index]

    # Keep match with higher score when IoU is bigger than threshold.
    boxes = matches_sorted[:, :4]
    ious = box_iou_batch(boxes, boxes)
    ious = ious - np.eye(rows)

    keep = np.ones(rows, dtype=bool)
    for index, iou in enumerate(ious):
        if not keep[index]:
            continue

        condition = iou > iou_threshold
        keep = keep & ~condition

    return matches_sorted[keep]


def annotate_img(img, xyxy, text, font=cv2.FONT_HERSHEY_SIMPLEX):
    x1, y1, x2, y2 = xyxy

    thickness = int(max(img.shape) / 180)
    cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=thickness)

    scale = thickness / 4
    text_width, text_height = cv2.getTextSize(
        text=text,
        fontFace=font,
        fontScale=scale,
        thickness=thickness,
    )[0]

    text_x = x1 + thickness
    text_y = y1 - thickness

    text_background_x1 = x1
    text_background_y1 = y1 - 2 * thickness - text_height

    text_background_x2 = x1 + 2 * thickness + text_width
    text_background_y2 = y1

    cv2.rectangle(
        img=img,
        pt1=(text_background_x1, text_background_y1),
        pt2=(text_background_x2, text_background_y2),
        color=(0, 255, 0),
        thickness=cv2.FILLED,
    )
    cv2.putText(
        img=img,
        text=text,
        org=(text_x, text_y),
        fontFace=font,
        fontScale=scale,
        color=(255, 255, 255),
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
