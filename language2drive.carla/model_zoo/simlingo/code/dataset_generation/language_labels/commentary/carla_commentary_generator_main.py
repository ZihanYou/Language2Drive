import argparse
from carla_commentary_generator import COMsGenerator
import string
import random
import pathlib
import json
import os
from tqdm import tqdm

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def parse_arguments():
    parser = argparse.ArgumentParser(description="QA Generator for DriveLM Carla")

    # Dataset and path settings
    path_group = parser.add_argument_group('Dataset and Path Settings')
    # path_group.add_argument('--base-folder', type=str, default='database',
    #                         help='Base folder for dataset')
    path_group.add_argument('--path-keyframes', type=str, default='path/to/keyframes.txt',
                            help='Path to the keyframes.txt')
    path_group.add_argument('--data-directory', type=str, default='database/simlingo',
                            help='Data directory containing the dataset')
    path_group.add_argument('--output-directory', type=str, default='database/simlingo/commentary',
                            help='Output directory for the vqa-graph')
    path_group.add_argument('--output-examples-directory', type=str, default='database/simlingo/commentary',
                            help='Output directory for examples of the vqa-graph')

    # Image and camera parameters
    img_group = parser.add_argument_group('Image and Camera Parameters')
    img_group.add_argument('--target-image-size', nargs=2, type=int, default=[1024, 358],
                           help='Target image size [width, height]')
    img_group.add_argument('--original-image-size', nargs=2, type=int, default=[1024, 512],
                           help='Original image size [width, height]')
    img_group.add_argument('--original-fov', type=float, default=110,
                           help='Original field of view')

    # Region of interest (ROI) for image projection
    roi_group = parser.add_argument_group('Region of Interest (ROI) Parameters')
    roi_group.add_argument('--min-y', type=int, default=0,
                           help='Minimum Y coordinate for ROI (to cut part of the bottom)')
    roi_group.add_argument('--max-y', type=int, default=358,
                           help='Maximum Y coordinate for ROI (to cut part of the bottom)')

    # Sampling parameters
    sampling_group = parser.add_argument_group('Sampling Parameters')
    sampling_group.add_argument('--random-subset-count', type=int, default=-1,
                                help='Number of random samples to use (-1 for all samples)')
    sampling_group.add_argument('--sample-frame-mode', choices=['all', 'keyframes', 'uniform'], default='all',
                                help='Frame sampling mode')
    sampling_group.add_argument('--sample-uniform-interval', type=int, default=1,
                                help='Interval for uniform sampling (if sample-frame-mode is "uniform")')

    # Visualization and saving options
    viz_group = parser.add_argument_group('Visualization and Saving Options')
    viz_group.add_argument('--save-examples', action='store_true', default=False,
                           help='Save example images')
    viz_group.add_argument('--visualize-projection', action='store_true', default=False,
                           help='Visualize object centers & bounding boxes in the image')
    viz_group.add_argument('--filter-routes-by-result', action='store_true', default=True,
                           help='Skip routes based on expert driving results')
    viz_group.add_argument('--skip-existing', action='store_true', default=True,
                           help='Skip existing files when saving examples')

    args = parser.parse_args()

    # Compute derived parameters
    args.min_x = args.original_image_size[0] // 2 - args.target_image_size[0] // 2
    args.max_x = args.original_image_size[0] // 2 + args.target_image_size[0] // 2
    if args.max_y is None:
        args.max_y = args.target_image_size[1]

    return args


if __name__ == '__main__':
    args = parse_arguments()
    multi_processing = True

    com_generator = COMsGenerator(args)
    len_data_boxes = len(com_generator.data_boxes_paths)
    if multi_processing:
        from tqdm.contrib.concurrent import process_map
        r = process_map(com_generator.create_commentary, range(0, len_data_boxes), max_workers=64, chunksize=10000)
    else:
        for i in tqdm(range(len_data_boxes)):
            com_generator.create_commentary(i)

    com_generator.save_stats()