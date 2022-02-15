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
pip install histomicstk 'large-image[openslide,ometiff,openjpeg,bioformats]' h5py 'tensorflow' --find-links https://girder.github.io/large_image_wheels

# Note: We also need to have the configuration file FeatureExtraction.xml in the
# same directory as the present executable file, FeatureExtraction.py.

# ############
# RUN
# ############

# Note: that path/to/myproject can be an absolute or relative path to a directory that
# must have subdirectories: svs, centroid.  The svs and centroid subdirectories have the
# inputs.  The outputs are HistomicsML_dataset.h5 and pca_model_sample.pkl in the
# myproject directory.

python FeatureExtraction.py --projectName path/to/myproject --superpixelSize 64 --patchSize 128

"""

import os
import json
import h5py
import time

import pandas as pd
import numpy as np

import histomicstk.preprocessing.color_normalization as htk_cnorm
import histomicstk.utils as htk_utils

import large_image

from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras import applications

from ctk_cli import CLIArgumentParser
from PIL import Image  # was: from scipy.misc import imresize
from sklearn.decomposition import PCA
import joblib  # was: from sklearn.externals import joblib

from histomicstk.cli import utils as cli_utils

import logging

logging.basicConfig(level=logging.CRITICAL)

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def compute_superpixel_data(
    model,
    slide_path,
    tile_position,
    centroid_path,
    args,
    superpixel_kwargs,
    src_mu_lab=None,
    src_sigma_lab=None,
):
    # get slide tile source
    ts = large_image.getTileSource(slide_path)

    # get scale for the tile and adjust centroids points
    ts_metadata = ts.getMetadata()

    f = h5py.File(centroid_path)
    x_centroids = f["x_centroid"][:]
    y_centroids = f["y_centroid"][:]

    # get requested tile
    tile_info = ts.getSingleTile(
        tile_position=tile_position,
        format=large_image.tilesource.TILE_FORMAT_NUMPY,
        **superpixel_kwargs
    )

    im_tile = tile_info["tile"][:, :, :3]

    # perform color normalization
    im_nmzd = htk_cnorm.reinhard(
        im_tile,
        args.reference_mu_lab,
        args.reference_std_lab,
        src_mu=src_mu_lab,
        src_sigma=src_sigma_lab,
    )

    im_height, im_width = im_nmzd.shape[:2]

    left = tile_info["gx"]
    top = tile_info["gy"]

    # get width and height
    width = tile_info["width"]
    height = tile_info["height"]

    n_superpixels = len(x_centroids)

    tile_features = []
    tile_x_centroids = []
    tile_y_centroids = []
    is_first = True

    for i in range(n_superpixels):
        if (
            left < x_centroids[i] <= left + width
            and top < y_centroids[i] <= top + height
        ):

            cen_x = x_centroids[i] - left
            cen_y = y_centroids[i] - top

            # get bounds of superpixel region
            min_row, max_row, min_col, max_col = get_patch_bounds(
                cen_y, cen_x, args.patchSize, im_height, im_width
            )

            # resize superpixel patch
            """
            im_patch = imresize(
                im_nmzd[min_row:max_row, min_col:max_col, :],
                (args.patchSizeResized, args.patchSizeResized, 3),
            )
            """
            im_patch = np.array(
                Image.fromarray(im_nmzd[min_row:max_row, min_col:max_col, :]).resize(
                    (args.patchSizeResized, args.patchSizeResized)
                )
            )

            # get superpixel features
            fcn = model.predict(
                preprocess_input(np.expand_dims(image.img_to_array(im_patch), axis=0))
            )

            if is_first:
                tile_features = fcn
                is_first = False

            else:
                tile_features = np.append(tile_features, fcn, axis=0)

            tile_x_centroids.append(x_centroids[i])
            tile_y_centroids.append(y_centroids[i])

    return tile_features, tile_x_centroids, tile_y_centroids


def compute_superpixel_data_pca(
    model,
    pca,
    slide_path,
    tile_position,
    centroid_path,
    args,
    superpixel_kwargs,
    src_mu_lab=None,
    src_sigma_lab=None,
):
    # get slide tile source
    ts = large_image.getTileSource(slide_path)

    # get scale for the tile and adjust centroids points
    ts_metadata = ts.getMetadata()

    f = h5py.File(centroid_path)
    x_centroids = f["x_centroid"][:]
    y_centroids = f["y_centroid"][:]

    # get requested tile
    tile_info = ts.getSingleTile(
        tile_position=tile_position,
        format=large_image.tilesource.TILE_FORMAT_NUMPY,
        **superpixel_kwargs
    )

    im_tile = tile_info["tile"][:, :, :3]

    # perform color normalization
    im_nmzd = htk_cnorm.reinhard(
        im_tile,
        args.reference_mu_lab,
        args.reference_std_lab,
        src_mu=src_mu_lab,
        src_sigma=src_sigma_lab,
    )

    im_height, im_width = im_nmzd.shape[:2]

    left = tile_info["gx"]
    top = tile_info["gy"]

    # get width and height
    width = tile_info["width"]
    height = tile_info["height"]

    n_superpixels = len(x_centroids)

    tile_features = []
    tile_x_centroids = []
    tile_y_centroids = []
    is_first = True

    for i in range(n_superpixels):
        if (
            left < x_centroids[i] <= left + width
            and top < y_centroids[i] <= top + height
        ):

            cen_x = x_centroids[i] - left
            cen_y = y_centroids[i] - top

            # get bounds of superpixel region
            min_row, max_row, min_col, max_col = get_patch_bounds(
                cen_y, cen_x, args.patchSize, im_height, im_width
            )

            # resize superpixel patch
            """
            im_patch = imresize(
                im_nmzd[min_row:max_row, min_col:max_col, :],
                (args.patchSizeResized, args.patchSizeResized, 3),
            )
            """
            im_patch = np.array(
                Image.fromarray(im_nmzd[min_row:max_row, min_col:max_col, :]).resize(
                    (args.patchSizeResized, args.patchSizeResized)
                )
            )

            # get superpixel features
            fcn = model.predict(
                preprocess_input(np.expand_dims(image.img_to_array(im_patch), axis=0))
            )

            # reduce the fcn features
            features = pca.transform(fcn)

            if is_first:
                tile_features = features
                is_first = False

            else:
                tile_features = np.append(tile_features, features, axis=0)

            tile_x_centroids.append(x_centroids[i])
            tile_y_centroids.append(y_centroids[i])

    return tile_features, tile_x_centroids, tile_y_centroids


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

    if args.inputPCAModel:
        print("\n>> Load PCA fitted model ... \n")
        pca = joblib.load(args.inputPCAModel)
    else:
        pca = "NULL"

    inputSlidePath = args.projectName + "/svs"
    inputCentroidPath = args.projectName + "/centroid"
    outputDataSet = args.projectName + "/HistomicsML_dataset.h5"
    outputPCAsample = args.projectName + "/pca_model_sample.pkl"

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
    # Check whether centroid directory exists
    #
    if os.path.isdir(inputCentroidPath):
        centroid_paths = [
            os.path.join(inputCentroidPath, files)
            for files in os.listdir(inputCentroidPath)
            if os.path.isfile(os.path.join(inputCentroidPath, files))
        ]
    else:
        raise IOError("Centroid path is not directory.")

    print("\n>> Reading VGG pre-trained model ... \n")
    model = applications.vgg16.VGG16(include_top=True, weights="imagenet")
    model = Model(inputs=model.input, outputs=model.get_layer("fc1").output)

    print("Generate train dataset ... ")
    slide_superpixel_data = []
    slide_x_centroids = []
    slide_y_centroids = []
    slide_name_list = []
    slide_superpixel_index = []

    total_n_slides = len(img_paths)
    first_superpixel_index = np.zeros((total_n_slides, 1), dtype=np.int32)
    slide_wsi_mean = np.zeros((total_n_slides, 3), dtype=np.float32)
    slide_wsi_stddev = np.zeros((total_n_slides, 3), dtype=np.float32)

    total_n_superpixels = 0

    index = 0

    for i in range(len(img_paths)):

        src_mu_lab = None
        src_sigma_lab = None

        for j in range(len(centroid_paths)):

            centroid_name = centroid_paths[j].split("/")[-1].split(".")[0]
            slide_name = img_paths[i].split("/")[-1].split(".")[0]

            if slide_name == centroid_name:

                slide_name_list.append(slide_name)

                #
                # Read Input Image
                #
                print("\n>> Reading input image ... \n")

                print("{} is processing ... \n".format(slide_name))

                ts = large_image.getTileSource(img_paths[i])

                ts_metadata = ts.getMetadata()
                scale = ts_metadata["magnification"] / args.max_mag

                superpixel_mag = args.max_mag * scale
                superpixel_tile_size = args.max_tile_size * scale

                print(json.dumps(ts_metadata, indent=2))

                is_wsi = ts_metadata["magnification"] is not None

                if is_wsi:

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
                        "tile_size": {"width": superpixel_tile_size},
                        "scale": {"magnification": superpixel_mag},
                    }

                    start_time = time.time()

                    num_tiles = ts.getSingleTile(**it_kwargs)["iterator_range"][
                        "position"
                    ]

                    print("Number of tiles = {}".format(num_tiles))

                    tile_fgnd_frac_list = htk_utils.compute_tile_foreground_fraction(
                        img_paths[i], im_fgnd_mask_lres, fgnd_seg_scale, it_kwargs
                    )

                    num_fgnd_tiles = np.count_nonzero(
                        tile_fgnd_frac_list >= args.min_fgnd_frac
                    )

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
                        img_paths[i], 0.01, magnification=superpixel_mag
                    )

                    rstats_time = time.time() - start_time

                    print(
                        "Reinhard stats computation time = {}".format(
                            cli_utils.disp_time_hms(rstats_time)
                        )
                    )

                    print("\n>> Detecting superpixel data ...\n")

                    superpixel_data = []
                    superpixel_x_centroids = []
                    superpixel_y_centroids = []

                    is_first = True

                    for tile in ts.tileIterator(**it_kwargs):
                        tile_position = tile["tile_position"]["position"]

                        if tile_fgnd_frac_list[tile_position] <= args.min_fgnd_frac:
                            continue

                        # detect superpixel data
                        if args.inputPCAModel:
                            (
                                tile_features,
                                tile_x_centroids,
                                tile_y_centroids,
                            ) = compute_superpixel_data_pca(
                                model,
                                pca,
                                img_paths[i],
                                tile_position,
                                centroid_paths[j],
                                args,
                                it_kwargs,
                                src_mu_lab,
                                src_sigma_lab,
                            )
                        else:
                            (
                                tile_features,
                                tile_x_centroids,
                                tile_y_centroids,
                            ) = compute_superpixel_data(
                                model,
                                img_paths[i],
                                tile_position,
                                centroid_paths[j],
                                args,
                                it_kwargs,
                                src_mu_lab,
                                src_sigma_lab,
                            )

                        print("tile_position = {}".format(tile_position))

                        if len(tile_features) > 0:

                            if is_first:
                                superpixel_data = tile_features
                                is_first = False
                            else:
                                superpixel_data = np.append(
                                    superpixel_data, tile_features, axis=0
                                )

                            superpixel_x_centroids.extend(tile_x_centroids)
                            superpixel_y_centroids.extend(tile_y_centroids)

                    n_superpixels = len(superpixel_x_centroids)
                    x_centroids = np.asarray(
                        superpixel_x_centroids, dtype=np.float32
                    ).reshape((n_superpixels, 1))
                    y_centroids = np.asarray(
                        superpixel_y_centroids, dtype=np.float32
                    ).reshape((n_superpixels, 1))

                    first_superpixel_index[index, 0] = total_n_superpixels
                    slide_superpixel_data = (
                        superpixel_data
                        if index == 0
                        else np.append(slide_superpixel_data, superpixel_data, axis=0)
                    )
                    slide_x_centroids = (
                        x_centroids
                        if index == 0
                        else np.append(slide_x_centroids, x_centroids, axis=0)
                    )
                    slide_y_centroids = (
                        y_centroids
                        if index == 0
                        else np.append(slide_y_centroids, y_centroids, axis=0)
                    )

                    slide_wsi_mean[index] = src_mu_lab
                    slide_wsi_stddev[index] = src_sigma_lab
                    slide_index = np.zeros((n_superpixels, 1), dtype=np.int32)
                    slide_index.fill(index)
                    slide_superpixel_index = (
                        slide_index
                        if index == 0
                        else np.append(slide_superpixel_index, slide_index, axis=0)
                    )
                    total_n_superpixels += n_superpixels
                    index += 1

    if args.inputPCAModel:
        superpixel_feature_map = np.asarray(slide_superpixel_data, dtype=np.float32)
    else:
        print("Fitting PCA ... ")
        df = pd.DataFrame(
            data=slide_superpixel_data, columns=[_ for _ in range(args.fcn)]
        )
        df_sample = df.reindex(np.random.permutation(df.index)).sample(
            frac=args.pca_sample_scale
        )

        pca = PCA(n_components=args.pca_dim)
        pca.fit(df_sample.values)
        joblib.dump(pca, outputPCAsample)

        superpixel_feature_map = np.asarray(
            pca.transform(slide_superpixel_data), dtype=np.float32
        )

    slide_x_centroids = np.asarray(slide_x_centroids, dtype=np.float32)
    slide_y_centroids = np.asarray(slide_y_centroids, dtype=np.float32)

    # get mean and standard deviation for train
    slide_feature_mean = np.reshape(
        np.mean(superpixel_feature_map[:], axis=0), (superpixel_feature_map.shape[1], 1)
    ).astype(np.float32)

    slide_feature_stddev = np.reshape(
        np.std(superpixel_feature_map[:], axis=0), (superpixel_feature_map.shape[1], 1)
    ).astype(np.float32)

    total_time_taken = time.time() - total_start_time

    print("Total analysis time = {}".format(cli_utils.disp_time_hms(total_time_taken)))

    print(">> Writing raw H5 data file")
    output = h5py.File(outputDataSet, "w")
    output.create_dataset("slides", data=slide_name_list)
    output.create_dataset("slideIdx", data=slide_superpixel_index)
    output.create_dataset("dataIdx", data=first_superpixel_index)
    output.create_dataset("mean", data=slide_feature_mean)
    output.create_dataset("std_dev", data=slide_feature_stddev)
    output.create_dataset("features", data=superpixel_feature_map)
    output.create_dataset("x_centroid", data=slide_x_centroids)
    output.create_dataset("y_centroid", data=slide_y_centroids)
    output.create_dataset("wsi_mean", data=slide_wsi_mean)
    output.create_dataset("wsi_stddev", data=slide_wsi_stddev)
    output.create_dataset("patch_size", data=args.superpixelSize)
    output.close()


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
