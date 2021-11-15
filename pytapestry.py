# Copyright (c) 2021 Ewan Short
# License: MIT

""" A script for creating tapestry templates. Note if provided, the
--tapestry_width_cells argument takes precedence over the --tapestry_width
argument. Happy 40th Birthday Zoe!
"""

import numpy as np
from pathlib import Path
import argparse

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
