# Copyright (c) 2021 Ewan Short
# License: MIT

""" A script for creating tapestry templates. Note if provided, the
--tapestry_width_cells argument takes precedence over the --tapestry_width
argument. Happy 40th Birthday Zoe!
"""

import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

from skimage.io import imread, imsave
from skimage.transform import rescale
from sklearn.metrics import pairwise_distances_argmin
from sklearn.cluster import KMeans
from sklearn.utils import shuffle


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "filepath", type=Path, help='full path to the image file.')
parser.add_argument(
    '--n-colours', '-c', type=int, default=10,
    help='number of colours to reduce image to')
parser.add_argument(
    '--canvas-count', '-cc', type=int, default=10,
    help='the mesh count per inch of the intended tapestry canvas')
parser.add_argument(
    '--tapestry-width', '-w', type=float, default=.25,
    help='the intended width of the tapestry in metres')
parser.add_argument(
    '--tapestry-width-cells', '-wp', type=int, default=0,
    help='the intended width of the tapestry in canvas cells (pixels)')
parser.add_argument(
    '--markers', '-m', type=int, default=0,
    help='cell interval for marker placement')

args = parser.parse_args()

screen_res = 96
pixel_scale = screen_res//args.canvas_count
if args.tapestry_width_cells == 0:
    tapestry_width = args.tapestry_width
else:
    tapestry_width = args.tapestry_width_cells/args.canvas_count*2.54/100

tap_im = imread(args.filepath)
height, width, depth = tap_im.shape
physical_width = width*2.54/(args.canvas_count*100)
width_ratio = tapestry_width/physical_width

tap_im = rescale(tap_im, width_ratio, multichannel=True)
tap_im = np.array(tap_im, dtype=np.float64)
height, width, depth = tap_im.shape

im_array = np.reshape(tap_im, (width * height, depth))

im_array_sample = shuffle(im_array, random_state=0, n_samples=1_000)
kmeans = KMeans(n_clusters=args.n_colours, random_state=0).fit(im_array_sample)
labels = kmeans.predict(im_array)

codebook_random = shuffle(im_array, random_state=0, n_samples=args.n_colours)
labels_random = pairwise_distances_argmin(codebook_random, im_array, axis=0)


def recreate_image(codebook, labels, width, height):
    return codebook[labels].reshape(height, width, -1)


im_reduced = recreate_image(kmeans.cluster_centers_, labels, width, height)
im_reduced_scaled = np.repeat(im_reduced, repeats=pixel_scale, axis=0)
im_reduced_scaled = np.repeat(im_reduced_scaled, repeats=pixel_scale, axis=1)

out_filename = args.filepath.stem + '_{}_{}_{}.png'.format(
    args.n_colours, args.canvas_count, int(np.floor(tapestry_width*1000)))
out_filepath = args.filepath.parent / out_filename

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import rcParams
import matplotlib.patheffects as pe

cols = 2
num_rows = np.ceil(len(kmeans.cluster_centers_) / cols)

fig, ax = plt.subplots()
ax.axis('off')
cell_height = .1
cell_width = .5
swatch_width = .1

ax.set_ylim([1-num_rows*cell_height,1])

for i, colour in enumerate(kmeans.cluster_centers_):
    # import pdb; pdb.set_trace()
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
            xy=(swatch_start_x, y-cell_height), width=swatch_width, height=.1,
            facecolor=colour))

plt.savefig(
    args.filepath.parent / 'swatch.png', dpi=96,
    bbox_inches='tight', format='png')


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


if args.markers > 0:
    im_reduced_scaled = add_markers(
        im_reduced_scaled, args.markers, pixel_scale)

imsave(out_filepath, (im_reduced_scaled*255).astype(np.uint8))

print('Template is {} cells high, and {} cells wide.'.format(
    im_reduced.shape[0], im_reduced.shape[1]))
print('Template image is saved as {}.'.format(out_filepath))
msg = 'To better reveal each cell, the template image has been '
msg += 'scaled by a factor of {}.'.format(pixel_scale)
print(msg)
msg = 'Tapestry cells thus correspond to {0} by {0} '.format(pixel_scale)
msg += 'blocks of pixels in the template image.'
print(msg)
print('Happy 40th Birthday Zoe!')
