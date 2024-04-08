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
    cmap="gray"
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
            axes[i].set_title(list_imgs_title[i], fontsize=int(figsize*2.5))
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()
