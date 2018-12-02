# encoding: utf-8


import os
import random
import shutil


IMG_PATH = './imgs'
NORMAL_IMG_PATH = './imgs/normal_img'
MALWARE_IMG_PATH = './imgs/malware_img'
DATA_PATH = './data'
TRAIN_PATH = './data/train'
TEST_PATH = './data/test'
TEST_SIZE = 0.2


def safe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':


    # 创建文件夹
    safe_mkdir(TRAIN_PATH)
    safe_mkdir(TEST_PATH)


    # 查找所有的图片
    normal_imgs = os.listdir(NORMAL_IMG_PATH)
    malware_imgs = os.listdir(MALWARE_IMG_PATH)


    # 乱序
    random.shuffle(normal_imgs)
    random.shuffle(malware_imgs)


    # 划分正常软件
    for img in normal_imgs[0: int((1-TEST_SIZE)*len(normal_imgs))]:
        source = os.path.join(NORMAL_IMG_PATH, img)
        target = os.path.join(TRAIN_PATH, img)
        shutil.copy(source, target)

    for img in normal_imgs[int((1-TEST_SIZE)*len(normal_imgs)):]:
        source = os.path.join(NORMAL_IMG_PATH, img)
        target = os.path.join(TEST_PATH, img)
        shutil.copy(source, target)


    # 划分恶意软件
    # 选 10000 张划入训练集
    # 选 2000 张划入测试集
    for img in malware_imgs[0: 10000]:
        source = os.path.join(MALWARE_IMG_PATH, img)
        target = os.path.join(TRAIN_PATH, img)
        shutil.copy(source, target)

    for img in malware_imgs[10000: 12000]:
        source = os.path.join(MALWARE_IMG_PATH, img)
        target = os.path.join(TEST_PATH, img)
        shutil.copy(source, target)

