#!/usr/bin/env python3
import os
import os.path as osp

from dataflow import MapData
from dataflow.dataflow.serialize import LMDBSerializer

from deepclr.data.datasets.kitti import KittiOdometryVelodyneData
from deepclr.data.transforms.transforms import SystematicErasing


SEQUENCES = ['00']
NTH = 2


def convert_sequence(base_path: str, sequence: str, output_file: str) -> None:
    # input
    df = KittiOdometryVelodyneData(base_path, sequence, shuffle=False)

    # transform
    transform = SystematicErasing(NTH)
    df = MapData(df, func=lambda x: transform(x))

    # output
    LMDBSerializer.save(df, output_file, write_frequency=5000)


def main():
    # get kitti paths
    indy_path = os.getenv('INDY_PATH')
    if indy_path is None:
        raise RuntimeError("Environment variable INDY_PATH not defined.")
    kitti_base_path = osp.join(indy_path, 'original')
    kitti_odometry_path = osp.join(indy_path, 'odometry')

    # create output directory
    os.makedirs(kitti_odometry_path, exist_ok=True)

    # iterate sequences
    for seq in SEQUENCES:
        print(f"Convert sequence {seq}")
        output_file = osp.join(kitti_odometry_path, f'{seq}.lmdb')
        convert_sequence(kitti_base_path, seq, output_file)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user")
