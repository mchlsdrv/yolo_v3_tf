import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.style.use('ggplot')


def draw_bboxes(image: np.ndarray, bboxes: list or np.ndarray):

    fig, ax = plt.subplots(figsize=(20, 20))
    for bbox in bboxes:
        # - Present the image in a figure
        ax.imshow(image, cmap='gray')

        # - Get the bbox confidence and the coordinates
        c, x, y, width, height = bbox

        # - Draw the current bbox
        rect = Rectangle((x, y), width=width, height=height, linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        ax.text(x=x, y=y, s=f'Yeast ({c:.2f})', c='g')

    return fig, ax


def plot_detections(images, bboxes, file_names: list, save_dir: pathlib.Path):

    os.makedirs(save_dir, exist_ok=True)

    for img, img_bboxes, fl_nm in zip(images, bboxes, file_names):
        fig, ax = draw_bboxes(image=img, bboxes=img_bboxes)
        fig.savefig(save_dir / f'{fl_nm}.png')
        plt.close(fig)


def plot(images, labels, suptitle='', figsize=(25, 10), save_file: pathlib.Path = None) -> None:
    fig, ax = plt.subplots(1, len(images), figsize=figsize)
    for idx, (img, lbl) in enumerate(zip(images, labels)):
        ax[idx].imshow(img, cmap='gray')
        ax[idx].set_title(lbl)

    fig.suptitle(suptitle)

    save_figure(figure=fig, save_file=save_file)


def save_figure(figure, save_file):
    if isinstance(save_file, pathlib.Path):
        os.makedirs(save_file.parent, exist_ok=True)
        figure.savefig(str(save_file))
        plt.close(figure)


def plot_image_histogram(images: np.ndarray, labels: list, n_bins: int = 256, figsize: tuple = (25, 50), density: bool = False, save_file: pathlib.Path = None):
    fig, ax = plt.subplots(2, len(images), figsize=figsize)
    for idx, (img, lbl) in enumerate(zip(images, labels)):

        vals, bins = np.histogram(img, n_bins, density=True)
        if density:
            vals = vals / vals.sum()
        vals, bins = vals[1:], bins[1:][:-1]  # don't include the 0

        # - If there is only a single plot - no second dimension will be available, and it will result in an error
        if len(images) > 1:
            hist_ax = ax[0, idx]
            img_ax = ax[1, idx]
        else:
            hist_ax = ax[0]
            img_ax = ax[1]

        # - Plot the histogram
        hist_ax.bar(bins, vals)
        hist_ax.set_title('Intensity Histogram')
        max_val = 255 if img.max() > 1 else 1
        hist_ax.set(xlim=(0, max_val), ylim=(0., 1.), yticks=np.arange(0., 1.1, .1), xlabel='I (Intensity)', ylabel='P(I)')

        # - Show the image
        img_ax.imshow(img, cmap='gray')
        img_ax.set_title(lbl)

    save_figure(figure=fig, save_file=save_file)
