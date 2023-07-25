#!/usr/bin/env python3
import argparse
import os
import os.path as osp
import warnings

import numpy as np
from deepclr.data.datasets import mulran2kitti

from deepclr.evaluation import Evaluator
from deepclr.data.datasets.mulran import velo2cam


SEQUENCES = ['KAIST03']


def mat_to_vec(m: np.ndarray) -> np.ndarray:
    return m.reshape(1, 16)[0, :12]


def convert_poses(evaluator: Evaluator, mulran_base_path: str, sequence_name: str, output_dir: str) -> None:
    # load mulran calib
    mulran = mulran2kitti.odometry(mulran_base_path, sequence_name)
    calib = mulran.calib.T_cam0_velo

    # iterate predicted poses
    sequence = evaluator.get_sequence(sequence_name)
    kitti_poses = [mat_to_vec(velo2cam(pose, calib))
                   for pose in sequence.prediction.poses]

    # save poses
    np.savetxt(osp.join(output_dir, f'{sequence_name}.txt'), np.array(kitti_poses))


def main():
    # parse inputs
    parser = argparse.ArgumentParser(description="Export predicted transformations as KITTI poses.")
    parser.add_argument('input_path', type=str, help="path with predicted transformations")
    args = parser.parse_args()

    # get mulran base path
    mulran_path = os.getenv('MULRAN_PATH')
    if mulran_path is None:
        raise RuntimeError("Environment variable MULRAN_PATH not defined.")
    mulran_base_path = osp.join(mulran_path)

    # load input files
    evaluator = Evaluator.read(args.input_path)

    # create output path
    output_dir = osp.join(args.input_path, 'mulran')
    os.makedirs(output_dir, exist_ok=True)

    # iterate sequences
    sequence_found = False
    for seq in SEQUENCES:
        # check sequence
        if not evaluator.has_sequence(seq):
            continue
        sequence_found = True

        # convert poses
        convert_poses(evaluator, mulran_base_path, seq, output_dir)

    # warning if no sequence was found
    if not sequence_found:
        warnings.warn("No sequence found in input directory.")


if __name__ == '__main__':
    main()
