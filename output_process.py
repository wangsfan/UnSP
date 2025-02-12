# -*- coding: utf-8 -*-
"""
@File  : output_process.py
@Author: 林嚣张
@Date  : 2023/12/5 21:02
@Software  : pycharm
"""
import glob

import pandas as pd


def combine_csv(files):
    """
    将多个csv文件整合成一个DataFrame。
    :param files: 要整合的文件名列表。
    :return: 整合后的DataFrame。
    """

    first_column = pd.read_csv(files[0]).iloc[:, 0:1]
    for file in files:
        df = pd.read_csv(file).iloc[:, 1:]
        combined_df = pd.concat([first_column, df], axis=1)
        first_column = combined_df
    return combined_df

for loss in ['mse', 'lpips', 'ssim']:
    loss_path = glob.glob(f'saved_output/sampling_mode/fixed_nums_*/{loss}.csv')
    combined_df = combine_csv(loss_path)
    combined_df.to_csv(f'data_csv/sampling_mode_fixed_nums_{loss}.csv')
