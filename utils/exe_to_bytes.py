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
MAX_PROCESS_NUM = 2  # 最大进程数


# 如果文件夹不存在则创建对应的文件夹
if not os.path.exists(NORMAL_BYTE_PATH):
    os.makedirs(NORMAL_BYTE_PATH)
if not os.path.exists(MALWARE_BYTE_PATH):
    os.makedirs(MALWARE_BYTE_PATH)
# if not os.path.exists(TEST_BYTE_PATH):
    # os.makedirs(TEST_BYTE_PATH)


def exe_to_bytes(exe_file, byte_file):
    """
    二进制文件 => 字节码序列文件
    """
    bytes = []
    f_input = open(exe_file, 'rb')
    f_input.seek(0, 0)
    while True:
        byte = f_input.read(1)
        if len(byte) == 0:
            break
        else:
            bytes.append('%02X' % ord(byte))
    f_input.close()
    f_output = open(byte_file, 'w+')
    count = 0
    lenth = len(bytes)
    for byte in bytes:
        count += 1
        if count != lenth:
            f_output.write(byte + ' ')
        else:
            f_output.write(byte)
        if count % 16 == 0:
            f_output.write('\n')
    f_output.close() 


def get_byte_files(exe_path, byte_path):
    files = os.listdir(exe_path)
    p = Pool(MAX_PROCESS_NUM)  # windows 下必须放在 “if __name__ == '__main__':” 后面，否则会报错
    for file in files:
        if os.path.exists(os.path.join(byte_path, file + '.txt')):  # 跳过已有的文件
            continue
        exe_file = os.path.join(exe_path, file)
        byte_file = os.path.join(byte_path, file + '.txt')
        p.apply_async(exe_to_bytes, args=(exe_file, byte_file,))
    p.close()
    p.join()


if __name__ == '__main__':
    get_byte_files(NORMAL_EXE_PATH, NORMAL_BYTE_PATH)
    get_byte_files(MALWARE_EXE_PATH, MALWARE_BYTE_PATH)
