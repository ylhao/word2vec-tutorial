# coding: utf-8


import os
import time
from multiprocessing import Pool


NORMAL_EXE_PATH = 'D:/malware_detection_dataset/normal'
MALWARE_EXE_PATH = 'D:/malware_detection_dataset/malware'
NORMAL_BYTE_PATH = 'D:/malware_detection_dataset/normal_byte'
MALWARE_BYTE_PATH = 'D:/malware_detection_dataset/malware_byte'
# TEST_EXE_PATH = 'D:/malware_detection_dataset/test'
# TEST_BYTE_PATH = 'D:/malware_detection_dataset/test_byte'


# 如果文件夹不存在则创建对应的文件夹
if not os.path.exists(NORMAL_BYTE_PATH):
    os.makedirs(NORMAL_BYTE_PATH)
if not os.path.exists(MALWARE_BYTE_PATH):
    os.makedirs(MALWARE_BYTE_PATH)
# if not os.path.exists(TEST_BYTE_PATH):
    # os.makedirs(TEST_BYTE_PATH)


def exe_to_bytes(exe_file):
    """
    二进制文件 => 字节码序列文件
    """
	with open(filename, 'rb') as f:
        content = f.read()
    hexst = binascii.hexlify(content)  # 将二进制文件转换为十六进制字符串
    fh = np.array([int(hexst[i:i+2], 16) for i in range(0, len(hexst), 2)])  # 按字节分割
    rn = len(fh)//width
    fh = np.reshape(fh[:rn*width],(-1, width))  # 根据设定的宽度生成矩阵
    return fh
	
if __name__ == '__main__':
    get_byte_files(NORMAL_EXE_PATH, NORMAL_BYTE_PATH)
    get_byte_files(MALWARE_EXE_PATH, MALWARE_BYTE_PATH)
