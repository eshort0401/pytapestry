
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.patheffects as pe


def create_swatch(colours, out_dir, filename):

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

        rcParams.update({'font.size': 10})
        rcParams.update({'font.weight': 'normal'})
        rcParams['font.family'] = 'serif'
        rcParams.update({'font.serif': 'Times New Roman'})

        colour_rgb = np.round(colour*255).astype(int)
        label = 'RGB = ({}, {}, {})'.format(
            colour_rgb[0], colour_rgb[1], colour_rgb[2])

        ax.text(
            swatch_start_x+swatch_width/3, y-2*cell_height/3, str(i+1),
            fontsize=12, zorder=1, fontweight='bold', color='w',
            path_effects=[
                pe.Stroke(linewidth=3, foreground='k'), pe.Normal()])

        ax.text(
            text_pos_x, y-cell_height/2, label, fontsize=10,
            horizontalalignment='left', verticalalignment='center')

        ax.add_patch(
            Rectangle(
                xy=(swatch_start_x, y-cell_height), width=swatch_width,
                height=.1, facecolor=colour))

    plt.savefig(
        out_dir / filename, dpi=96, bbox_inches='tight', format='png')


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


def recreate_image(codebook, labels, width, height):
    return codebook[labels].reshape(height, width, -1)
