# encoding: utf-8


import os
import random
import shutil


IMG_PATH = './imgs'
NORMAL_IMG_PATH = './imgs/normal_img'
MALWARE_IMG_PATH = './imgs/malware_img'
DATA_PATH = './data'
TRAIN_PATH = './data/train'
VALID_PATH = './data/valid'
TEST_PATH = './data/test'
NORMAL_TRAIN_PATH = './data/train/normal'
MALWARE_TRAIN_PATH = './data/train/malware'
NORMAL_VALID_PATH = './data/valid/normal'
MALWARE_VALID_PATH = './data/valid/malware'


def safe_mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


if __name__ == '__main__':


    # 创建文件夹
    safe_mkdir(NORMAL_TRAIN_PATH)
    safe_mkdir(NORMAL_VALID_PATH)
    safe_mkdir(MALWARE_TRAIN_PATH)
    safe_mkdir(MALWARE_VALID_PATH)


    # 查找所有的图片
    normal_imgs = os.listdir(NORMAL_IMG_PATH)
    malware_imgs = os.listdir(MALWARE_IMG_PATH)


    # 乱序
    random.shuffle(normal_imgs)
    random.shuffle(malware_imgs)


    # 划分正常软件
    # 80% 划入训练集
    # 20% 划入验证集
    for img in normal_imgs[0: int(0.8*len(normal_imgs))]:
        source = os.path.join(NORMAL_IMG_PATH, img)
        target = os.path.join(NORMAL_TRAIN_PATH, img)
        shutil.copy(source, target)


    for img in normal_imgs[int(0.8*len(normal_imgs)):]:
        source = os.path.join(NORMAL_IMG_PATH, img)
        target = os.path.join(NORMAL_VALID_PATH, img)
        shutil.copy(source, target)


    # 划分恶意软件
    # 选 10000 张划入训练集
    # 选 2000 张划入验证集
    for img in malware_imgs[0: 10000]:
        source = os.path.join(MALWARE_IMG_PATH, img)
        target = os.path.join(MALWARE_TRAIN_PATH, img)
        shutil.copy(source, target)


    for img in malware_imgs[10000: 12000]:
        source = os.path.join(MALWARE_IMG_PATH, img)
        target = os.path.join(MALWARE_VALID_PATH, img)
        shutil.copy(source, target)
