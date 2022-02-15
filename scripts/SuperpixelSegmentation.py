"""
# ############
# INSTALLATION
# ############

# Install memcached if it is not already installed.
for package in memcached; do
    if [ $(dpkg-query -W -f='${Status}' "${package}" 2>/dev/null | grep -c "ok installed") -eq 0 ]; then
        sudo apt install "${package}"
        # Note: If "sudo apt install memcached" gives "invoke-rc.d: policy-rc.d denied
        # execution of start." then update /usr/sbin/policy-rc.d with
        #     sudo printf '#!/bin/sh\nexit 0' > /usr/sbin/policy-rc.d
        # Then "sudo apt purge memcached" and try the install again.
    fi
done

# Create, launch, and update a Python virtual environment as ~/venv/superpixel (or
# elsewhere).
python -m venv ~/venv/superpixel && source ~/venv/superpixel/bin/activate && pip install -U pip setuptools wheel

# Install packages into our Python virtual environment.
pip install histomicstk 'large-image[openslide,ometiff,openjpeg,bioformats]' h5py --find-links https://girder.github.io/large_image_wheels

# Note: We also need to have the configuration file SuperpixelSegmentation.xml in the
# same directory as the present executable file, SuperpixelSegmentation.py.

# ############
# RUN
# ############

# Note: that path/to/myproject can be an absolute or relative path to a directory that
# must have subdirectories: svs, centroid, boundary.  The svs subdirectory has the input
# image that we want superpixels for.  The centroid and boundary subdirectories have the
# output.

python SuperpixelSegmentation.py --projectName path/to/myproject --superpixelSize 64 --patchSize 128

"""

import os
import json
import time
import h5py

import numpy as np
import dask

import histomicstk.preprocessing.color_normalization as htk_cnorm
import histomicstk.segmentation as htk_seg
import histomicstk.utils as htk_utils

import large_image

from ctk_cli import CLIArgumentParser
from scipy import ndimage
from skimage.measure import regionprops
from skimage.segmentation import slic

from histomicstk.cli import utils as cli_utils

import logging

logging.basicConfig(level=logging.CRITICAL)


def compute_boundary_data(
    img_path, tile_position, args, it_kwargs, src_mu_lab=None, src_sigma_lab=None
):

    # get slide tile source
    ts = large_image.getTileSource(img_path)

    # get scale for the tile and adjust centroids points
    ts_metadata = ts.getMetadata()

    # get requested tile information
    tile_info = ts.getSingleTile(
        tile_position=tile_position,
        format=large_image.tilesource.TILE_FORMAT_NUMPY,
        **it_kwargs
    )

    # get global x and y positions, tile height and width
    left = tile_info["gx"]
    top = tile_info["gy"]

    im_tile = tile_info["tile"][:, :, :3]

    # perform color normalization
    im_nmzd = htk_cnorm.reinhard(
        im_tile,
        args.reference_mu_lab,
        args.reference_std_lab,
        src_mu=src_mu_lab,
        src_sigma=src_sigma_lab,
    )

    # get red and green channels
    im_red = im_nmzd[:, :, 0]
    im_green = im_nmzd[:, :, 1]

    resize_scale = ts_metadata["magnification"] / args.analysis_mag

    re_superpixelSize = int(args.superpixelSize / resize_scale)

    # get foreground mask using numpy.spacing, a generalization of EPS
    im_ratio = im_red / (im_green + np.spacing(1))
    im_fgnd_mask = im_ratio > args.rg_ratio_superpixel

    # compute superpixel number
    n_rows, n_cols = im_nmzd.shape[:2]

    n_superpixels = int((n_rows / re_superpixelSize) * (n_cols / re_superpixelSize))

    # get labels
    im_label = slic(im_nmzd, n_segments=n_superpixels, compactness=args.compactness) + 1

    region_props = regionprops(im_label)

    valid_superpixel = []

    # get valid superpixels
    for i in range(len(region_props)):

        # grab superpixel label mask
        lmask = (im_label[:, :] == region_props[i].label).astype(bool)

        if np.sum(im_fgnd_mask & lmask) > args.min_fgnd_superpixel:

            # get x, y centroids for superpixel
            cen_x, cen_y = region_props[i].centroid

            # get bounds of superpixel region
            min_row, max_row, min_col, max_col = get_patch_bounds(
                cen_x, cen_y, args.patchSize, n_rows, n_cols
            )

            # get variance of superpixel region
            im_superpixel = im_nmzd[min_row:max_row, min_col:max_col, :] / 255.0

            var = ndimage.variance(im_superpixel)

            if var < args.min_var_superpixel:
                continue

            valid_superpixel.append(region_props[i].label)

    # resize image to get boundary and centroids at high resolution
    im_label = ndimage.zoom(im_label, int(resize_scale), order=0)
    n_rows, n_cols = im_label.shape[:2]
    region_props = regionprops(im_label)

    # initialize boundary and centroids
    x_cent = []
    y_cent = []
    x_brs = []
    y_brs = []

    # get boundary and centroids
    for i in range(len(region_props)):

        if region_props[i].label in valid_superpixel:
            min_row, max_row, min_col, max_col = get_boundary_bounds(
                region_props[i].bbox, 0, n_rows, n_cols
            )

            # grab label mask
            lmask = (
                im_label[min_row:max_row, min_col:max_col] == region_props[i].label
            ).astype(bool)

            mask = np.zeros((lmask.shape[0] + 2, lmask.shape[1] + 2), dtype=bool)
            mask[1:-1, 1:-1] = lmask

            # find boundaries
            bx, by = htk_seg.label.trace_object_boundaries(mask)
            bx = bx[0] + min_row
            by = by[0] + min_col

            with np.errstate(invalid="ignore"):
                # remove redundant points
                mby, mbx = htk_utils.merge_colinear(by.astype(float), bx.astype(float))

            # get superpixel boundary at highest-res
            y_brs.append(mbx + top)
            x_brs.append(mby + left)
            cen_x, cen_y = region_props[i].centroid

            # get superpixel centers at highest-res
            y_cent.append(round((cen_x + top), 1))
            x_cent.append(round((cen_y + left), 1))

    return x_cent, y_cent, x_brs, y_brs


def get_boundary_bounds(bbox, delta, m, n):
    min_row, min_col, max_row, max_col = bbox
    min_row_out = max(0, (min_row - delta))
    max_row_out = min(m - 1, (max_row + delta))
    min_col_out = max(0, (min_col - delta))
    max_col_out = min(n - 1, (max_col + delta))

    return min_row_out, max_row_out, min_col_out, max_col_out


def get_patch_bounds(cx, cy, patch, m, n):
    half_patch = patch / 2.0

    min_row = int(round(cx) - half_patch)
    max_row = int(round(cx) + half_patch)
    min_col = int(round(cy) - half_patch)
    max_col = int(round(cy) + half_patch)

    if min_row < 0:
        max_row = max_row - min_row
        min_row = 0

    if max_row > m - 1:
        min_row = min_row - (max_row - (m - 1))
        max_row = m - 1

    if min_col < 0:
        max_col = max_col - min_col
        min_col = 0

    if max_col > n - 1:
        min_col = min_col - (max_col - (n - 1))
        max_col = n - 1

    return min_row, max_row, min_col, max_col


def main(args):  # noqa: C901

    total_start_time = time.time()

    print("\n>> CLI Parameters ...\n")

    print(args)

    inputSlidePath = args.projectName + "/svs"
    outputCentroidDir = args.projectName + "/centroid"
    outputBoundaryDir = args.projectName + "/boundary"

    #
    # Check whether slide directory exists
    #
    if os.path.isdir(inputSlidePath):
        img_paths = [
            os.path.join(inputSlidePath, files)
            for files in os.listdir(inputSlidePath)
            if os.path.isfile(os.path.join(inputSlidePath, files))
        ]
    else:
        raise IOError("Slide path is not directory.")

    #
    # Initiate Dask client
    #
    print("\n>> Creating Dask client ...\n")

    start_time = time.time()

    c = cli_utils.create_dask_client(args)

    print(c)

    dask_setup_time = time.time() - start_time
    print("Dask setup time = {}".format(cli_utils.disp_time_hms(dask_setup_time)))

    for i in range(len(img_paths)):

        slide_name = img_paths[i].split("/")[-1].split(".")[0]

        src_mu_lab = None
        src_sigma_lab = None

        #
        # Read Input Image
        #
        print("\n>> Reading input image ... \n")

        ts = large_image.getTileSource(img_paths[i])

        ts_metadata = ts.getMetadata()

        print(json.dumps(ts_metadata, indent=2))

        if ts_metadata["magnification"] is not None:

            #
            # Compute tissue/foreground mask at low-res for whole slide images
            #
            print("\n>> Computing tissue/foreground mask at low-res ...\n")

            start_time = time.time()

            (
                im_fgnd_mask_lres,
                fgnd_seg_scale,
            ) = cli_utils.segment_wsi_foreground_at_low_res(ts)

            fgnd_time = time.time() - start_time

            print(
                "low-res foreground mask computation time = {}".format(
                    cli_utils.disp_time_hms(fgnd_time)
                )
            )

            it_kwargs = {
                "tile_size": {"width": args.analysis_tile_size},
                "scale": {"magnification": args.analysis_mag},
            }

            #
            # Compute foreground fraction of tiles in parallel using Dask
            #
            print("\n>> Computing foreground fraction of all tiles ...\n")

            start_time = time.time()

            num_tiles = ts.getSingleTile(**it_kwargs)["iterator_range"]["position"]

            print("Number of tiles = {}".format(num_tiles))

            tile_fgnd_frac_list = htk_utils.compute_tile_foreground_fraction(
                img_paths[i], im_fgnd_mask_lres, fgnd_seg_scale, it_kwargs
            )

            num_fgnd_tiles = np.count_nonzero(tile_fgnd_frac_list >= args.min_fgnd_frac)

            percent_fgnd_tiles = 100.0 * num_fgnd_tiles / num_tiles

            fgnd_frac_comp_time = time.time() - start_time

            print(
                "Number of foreground tiles = {0:d} ({1:2f}%%)".format(
                    num_fgnd_tiles, percent_fgnd_tiles
                )
            )

            print(
                "Tile foreground fraction computation time = {}".format(
                    cli_utils.disp_time_hms(fgnd_frac_comp_time)
                )
            )

            print("\n>> Computing reinhard color normalization stats ...\n")

            start_time = time.time()

            src_mu_lab, src_sigma_lab = htk_cnorm.reinhard_stats(
                img_paths[i], 0.01, magnification=args.analysis_mag
            )

            rstats_time = time.time() - start_time

            print(
                "Reinhard stats computation time = {}".format(
                    cli_utils.disp_time_hms(rstats_time)
                )
            )

            #
            # Detect boundary and centroids in parallel using Dask
            #
            print("\n>> Detecting boundary and centroids ...\n")

            start_time = time.time()

            tile_result_list = []

            for tile in ts.tileIterator(**it_kwargs):

                tile_position = tile["tile_position"]["position"]

                if tile_fgnd_frac_list[tile_position] <= args.min_fgnd_frac:
                    continue

                # detect superpixel data
                cur_result = dask.delayed(compute_boundary_data)(
                    img_paths[i],
                    tile_position,
                    args,
                    it_kwargs,
                    src_mu_lab,
                    src_sigma_lab,
                )

                # append result to list
                tile_result_list.append(cur_result)

            tile_result_list = dask.delayed(tile_result_list).compute()

            x_centroids = []
            y_centroids = []
            x_boundaries = []
            y_boundaries = []

            for x_cent, y_cent, x_brs, y_brs in tile_result_list:

                for x_c in x_cent:
                    x_centroids.append(x_c)

                for y_c in y_cent:
                    y_centroids.append(y_c)

                for x_b in x_brs:
                    x_boundaries.append(x_b)

                for y_b in y_brs:
                    y_boundaries.append(y_b)

            #
            # Create Centroids
            #
            print(">> Writing Slide data file")

            centroid_output = h5py.File(
                outputCentroidDir + "/" + slide_name + ".h5", "w"
            )
            centroid_output.create_dataset("x_centroid", data=x_centroids)
            centroid_output.create_dataset("y_centroid", data=y_centroids)
            centroid_output.close()

            boundary_out = open(outputBoundaryDir + "/" + slide_name + ".txt", "w")

            for j in range(len(x_centroids)):
                boundary_out.write("%s\t" % slide_name)
                boundary_out.write("%.1f\t" % x_centroids[j])
                boundary_out.write("%.1f\t" % y_centroids[j])

                for k in range(len(x_boundaries[j])):
                    boundary_out.write(
                        "%d,%d " % (x_boundaries[j][k], y_boundaries[j][k])
                    )

                boundary_out.write("\n")

            boundary_centroids_detection_time = time.time() - start_time

            boundary_out.close()

            print(
                "Boundary and centroids detection time = {}".format(
                    cli_utils.disp_time_hms(boundary_centroids_detection_time)
                )
            )

    total_time_taken = time.time() - total_start_time

    print("Total analysis time = {}".format(cli_utils.disp_time_hms(total_time_taken)))


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
