# coding: utf-8


import os
import time
import binascii
import numpy as np
from multiprocessing import Pool


NORMAL_EXE_PATH = 'D:/malware_detection_dataset/normal'
MALWARE_EXE_PATH = 'D:/malware_detection_dataset/malware'
NORMAL_BYTE_PATH = 'D:/malware_detection_dataset/normal_byte'
MALWARE_BYTE_PATH = 'D:/malware_detection_dataset/malware_byte'


# 如果文件夹不存在则创建对应的文件夹
if not os.path.exists(NORMAL_BYTE_PATH):
    os.makedirs(NORMAL_BYTE_PATH)
if not os.path.exists(MALWARE_BYTE_PATH):
    os.makedirs(MALWARE_BYTE_PATH)


def exe_to_bytes(exe_file, width):
    """
    二进制文件 => 字节码序列文件
    """
    with open(exe_file, 'rb') as f:
        content = f.read()
    hexst = binascii.hexlify(content)  # 将二进制文件转换为十六进制字符串
    fh = np.array([int(hexst[i:i+2], width) for i in range(0, len(hexst), 2)])  # 按字节分割
    # print(fh)
    rn = len(fh) // width
    fh = np.reshape(fh[:rn*width],(-1, width))  # 根据设定的宽度生成矩阵
    return fh


def get_normal_byte_files():
    files = os.listdir(NORMAL_EXE_PATH)
    for file in files:
        exe_file = os.path.join(NORMAL_EXE_PATH, file)
        print(exe_file)
        # 判断文件大小
        file_size = os.path.getsize(exe_file) / 1024 / 1024
        if file_size == 0:  # 有的文件可能为空
            continue
        # 正常软件有特大文件，这里需要过滤一下
        elif file_size > 10:  # 过滤掉大于 10 MB 的软件
            continue
        else:
            byte_file = os.path.join(NORMAL_BYTE_PATH, file + '.txt')
            fh = exe_to_bytes(exe_file, 16)
            np.savetxt(byte_file, fh, fmt='%d')


def get_malware_byte_files():
    files = os.listdir(MALWARE_EXE_PATH)
    for file in files:
        exe_file = os.path.join(MALWARE_EXE_PATH, file)
        print(exe_file)
        # 判断文件大小
        file_size = os.path.getsize(exe_file) / 1024 / 1024
        if file_size == 0:  # 有的文件可能为空
            continue
        # 目的是识别所有的恶意软件，所以恶意软件不能过滤
        # elif file_size > 10:  # 过滤掉大于 10 MB 的软件
            # continue
        else:
            byte_file = os.path.join(MALWARE_BYTE_PATH, file + '.txt')
            fh = exe_to_bytes(exe_file, 16)
            np.savetxt(byte_file, fh, fmt='%d')

	
if __name__ == '__main__':
    # get_normal_byte_files()
    get_malware_byte_files()
