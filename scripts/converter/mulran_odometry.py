#!/usr/bin/env python3
import os
import os.path as osp

from dataflow import MapData
from dataflow.dataflow.serialize import LMDBSerializer

from deepclr.data.datasets.mulran import MulranOdometryOusterData
from deepclr.data.transforms.transforms import SystematicErasing


SEQUENCES = ['DCC03', 'KAIST03', 'Sejong03', 'Riverside03']
NTH = 5


def convert_sequence(base_path: str, sequence: str, output_file: str) -> None:
    # input
    df = MulranOdometryOusterData(base_path, sequence, shuffle=False)

    # transform
    transform = SystematicErasing(NTH)
    df = MapData(df, func=lambda x: transform(x))

    # output
    LMDBSerializer.save(df, output_file, write_frequency=5000)


def main():
    # get kitti paths
    mulran_path = os.getenv('MULRAN_PATH')
    if mulran_path is None:
        raise RuntimeError("Environment variable MULRAN_PATH not defined.")
    mulran_base_path = osp.join(mulran_path)
    mulran_odometry_path = osp.join(mulran_path, 'odometry')

    # create output directory
    os.makedirs(mulran_odometry_path, exist_ok=True)

    # iterate sequences
    for seq in SEQUENCES:
        print(f"Convert sequence {seq}")
        output_file = osp.join(mulran_odometry_path, f'{seq}.lmdb')
        convert_sequence(mulran_base_path, seq, output_file)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user")
