# coding: utf-8


import os
import numpy as np
from PIL import Image
import binascii
import time



NORMAL_EXE_PATH = 'D:/malware_detection_dataset/normal'
MALWARE_EXE_PATH = 'D:/malware_detection_dataset/malware'
NORMAL_IMG_PATH = 'D:/malware_detection_dataset/normal_img'
MALWARE_IMG_PATH = 'D:/malware_detection_dataset/malware_img'
# TEST_EXE_PATH = 'D:/malware_detection_dataset/test'
# TEST_IMG_PATH = 'D:/malware_detection_dataset/test_img'
MAX_PROCESS_NUM = 2  # 最大进程数


# 如果文件夹不存在则创建对应的文件夹
if not os.path.exists(NORMAL_IMG_PATH):
    os.makedirs(NORMAL_IMG_PATH)
if not os.path.exists(MALWARE_IMG_PATH):
    os.makedirs(MALWARE_IMG_PATH)
# if not os.path.exists(TEST_IMG_PATH):
    # os.makedirs(TEST_IMG_PATH)


def getMatrixfrom_bin(filename, width):
    with open(filename, 'rb') as f:
        content = f.read()
    hexst = binascii.hexlify(content)  # 将二进制文件转换为十六进制字符串
    fh = np.array([int(hexst[i:i+2], 16) for i in range(0, len(hexst), 2)])  # 按字节分割
    rn = len(fh)//width
    fh = np.reshape(fh[:rn*width],(-1, width))  # 根据设定的宽度生成矩阵
    fh = np.uint8(fh)
    return fh


def get_img_files(exe_path, img_path):
    
    files = os.listdir(exe_path)
    print(files)
    for file in files:
        exe_file = os.path.join(exe_path, file)
        img_file = os.path.join(img_path, file + '.png')
        if os.path.exists(img_file):  # 判断是否已经生成过该 exe 文件对应的图片
            continue
        else:
            # 判断文件大小
            file_size = os.path.getsize(exe_file)
            file_size = file_size / 1024  # 转为 KB
            if file_size == 0:  # 有的文件可能为空
                continue
            elif file_size < 10:  # < 10 KB
                width = 32
            elif file_size < 30:  # < 30 KB
                width = 64
            elif file_size < 60:  # < 60 KB
                width = 128
            elif file_size < 100:  # < 100 KB
                width = 256
            elif file_size < 200:  # < 200 KB
                width = 384
            elif file_size < 500:  # < 500 KB
                width = 512
            elif file_size < 1000:  # < 1000 KB
                width = 768
            elif file_size < 10000:  # < 10000 KB
                width = 1024
            else:
                continue
            # 文件大小在合理范围内，执行以下语句
            print('{} ==> {}'.format(file, file + '.png'))
            fh = getMatrixfrom_bin(exe_file, width)
            im = Image.fromarray(fh)  # 转换为图像
            im.save(img_file)
            print('{} ==> {} done'.format(file, file + '.png'))


if __name__ == '__main__':
    start_time = time.time()
    print('start time:', start_time)
    # get_img_files(TEST_EXE_PATH, TEST_IMG_PATH)
    get_img_files(NORMAL_EXE_PATH, NORMAL_IMG_PATH)
    #get_img_files(MALWARE_EXE_PATH, MALWARE_IMG_PATH)
    end_time = time.time()
    print('end time:', end_time)
    print('time cost:', end_time - start_time)
