import shutil
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.patheffects as pe

from skimage.measure import label
from skimage.segmentation import expand_labels


def remove_isolated_pixels(labels_array, min_pixels):

    connected_regions = label(labels_array, connectivity=2, background=-10)
    lbl, counts = np.unique(connected_regions, return_counts=True)
    small_regions = lbl[counts < min_pixels]
    iteration = 0
    while small_regions.size > 0 and iteration < 5:
        print('Iteration {}'.format(iteration))
        for i in small_regions:
            region = (connected_regions == i).astype(int)
            boundary = expand_labels(region, distance=1.9) - region
            boundary_colours = labels_array[np.where(boundary)]
            mode = np.bincount(boundary_colours).argmax()
            labels_array[np.where(region)] = mode
        connected_regions = label(labels_array, connectivity=2, background=-10)
        lbl, counts = np.unique(connected_regions, return_counts=True)
        small_regions = lbl[counts < min_pixels]
        iteration += 1
        if iteration == 5:
            print('Maximum iterations reached. Algorithm did not converge.')

    return labels_array


def init_fonts(font_size=12):
    rcParams.update({'font.size': font_size})
    rcParams.update({'font.weight': 'normal'})
    rcParams['font.family'] = 'serif'
    rcParams.update({'font.serif': 'Times New Roman'})


def create_sub_templates(image, labels_array, markers, out_sub_dir):

    init_fonts()

    def label_colours(ax, sub_labels, x, y):
        # sub_labels = sub_labels[-1::-1, :]
        for i in range(sub_labels.shape[0]):
            for j in range(sub_labels.shape[1]):
                ax.text(
                    y[j]+.35, x[i]+.65, str(sub_labels[i, j]), fontsize=12,
                    zorder=1, fontweight='bold', color='w', path_effects=[
                        pe.Stroke(linewidth=3, foreground='k'), pe.Normal()])

    height, width, depth = image.shape
    x_blocks = height // markers + 1
    y_blocks = height // markers + 1
    for i in range(x_blocks):
        for j in range(y_blocks):
            fig, ax = plt.subplots()
            x_end = np.min([(i+1)*markers, height])
            y_end = np.min([(j+1)*markers, width])
            x = np.arange(i*markers, x_end+1)+0.5
            y = np.arange(j*markers, y_end+1)+0.5
            if len(x) < 2 or len(y) < 2:
                continue
            sub_image = image[i*markers:x_end, j*markers:y_end, :]
            sub_labels = labels_array[i*markers:x_end, j*markers:y_end]
            ax.imshow(sub_image, extent=(y[0], y[-1], x[-1], x[0]))
            label_colours(ax, sub_labels, x, y)
            plt.title('Sub-Template Row {} Column {}'.format(i, j))
            print(
                'Generating sub-template for row {}, column {}.'.format(i, j))
            plt.savefig(
                out_sub_dir / 'sub_template_row_{}_column_{}.png'.format(i, j),
                dpi=96, bbox_inches='tight', format='png')
            plt.close('all')


def create_swatch(colours, out_dir):

    cols = 2
    num_rows = np.ceil(len(colours) / cols)
    fig, ax = plt.subplots()
    ax.axis('off')
    cell_height = .1
    cell_width = .5
    swatch_width = .1
    ax.set_ylim([1-num_rows*cell_height, 1])

    for i, colour in enumerate(colours):

        row = i // cols
        col = i % cols
        y = 1 - row * cell_height
        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + .025
        init_fonts()
        colour_rgb = np.round(colour*255).astype(int)
        lbl = 'RGB = ({}, {}, {})'.format(
            colour_rgb[0], colour_rgb[1], colour_rgb[2])

        ax.text(
            swatch_start_x+swatch_width/3, y-2*cell_height/3, str(i+1),
            fontsize=12, zorder=1, fontweight='bold', color='w',
            path_effects=[
                pe.Stroke(linewidth=3, foreground='k'), pe.Normal()])

        ax.text(
            text_pos_x, y-cell_height/2, lbl, fontsize=10,
            horizontalalignment='left', verticalalignment='center')

        ax.add_patch(
            Rectangle(
                xy=(swatch_start_x, y-cell_height), width=swatch_width,
                height=.1, facecolor=colour))

    plt.savefig(
        out_dir / 'swatch.png', dpi=96, bbox_inches='tight', format='png')


def add_markers(im, interval, pixel_scale):
    height, width, depth = im.shape
    del_pix = interval*pixel_scale
    cross_len = pixel_scale
    cross_width = 2
    line_len = 1
    for i in range(1, height):
        for j in range(1, width):
            im[
                i*del_pix-cross_len:i*del_pix+cross_len,
                j*del_pix-cross_width:j*del_pix+cross_width, :] = 0
            im[
                i*del_pix-cross_width:i*del_pix+cross_width,
                j*del_pix-cross_len:j*del_pix+cross_len, :] = 0
            im[
                i*del_pix-cross_len+line_len:i*del_pix+cross_len-line_len,
                j*del_pix-cross_width+line_len:j*del_pix+cross_width-line_len,
                :] = 1
            im[
                i*del_pix-cross_width+line_len:i*del_pix+cross_width-line_len,
                j*del_pix-cross_len+line_len:j*del_pix+cross_len-line_len,
                :] = 1
    return im


def recreate_image(codebook, lbls, width, height):
    return codebook[lbls].reshape(height, width, -1)
