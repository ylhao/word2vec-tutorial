# coding: utf-8


import os
import numpy as np
from PIL import Image
import binascii
import time
import shutil


"""
只过滤良性软件即可
"""


NORMAL_EXE_PATH = 'D:/malware_detection_dataset/normal'
NORMAL_CLEAR_PATH = 'D:/malware_detection_dataset/normal_less_10MB'
# MALWARE_EXE_PATH = 'D:/malware_detection_dataset/malware'
# MALWARE_CLEAR_PATH = 'D:/malware_detection_dataset/malware_less_10MB'


if not os.path.exists(NORMAL_CLEAR_PATH):
    os.makedirs(NORMAL_CLEAR_PATH)
# if not os.path.exists(MALWARE_CLEAR_PATH):
#     os.makedirs(MALWARE_CLEAR_PATH)


def clear_exe(exe_path, clear_path):
    
    files = os.listdir(exe_path)
    for file in files:
        print(file)
        exe_file = os.path.join(exe_path, file)
        file_size = os.path.getsize(exe_file)  / 1024 / 1024  # 转为 MB
        if file_size == 0:  # 有的文件可能为空
            continue
        elif file_size < 10:  # < 10 MB
            # 复制文件
            to_file = os.path.join(clear_path, file)
            shutil.copy(exe_file, to_file)
        else:
            continue


if __name__ == '__main__':
    clear_exe(NORMAL_EXE_PATH, NORMAL_CLEAR_PATH)
