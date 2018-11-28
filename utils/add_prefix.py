# coding: utf-8


import os


# TEST_PATH = 'D:/malware_detection_dataset/test'
WINXP_PATH = 'D:/malware_detection_dataset/winxp'
WIN7_PATH = 'D:/malware_detection_dataset/win7'
WIN8_PATH = 'D:/malware_detection_dataset/win8'
WIN10_PATH = 'D:/malware_detection_dataset/win10'
# PREFIX_TEST = 'test_'
PREFIX_WINXP = 'winxp_'
PREFIX_WIN7 = 'win7_'
PREFIX_WIN8 = 'win8_'
PREFIX_WIN10 = 'win10_'


def add_prefix(prefix, path):
    """
    给一个文件夹下的所有的文件添加一个前缀
    """
    files = os.listdir(path)
    for file in files:
        os.rename(os.path.join(path, file), os.path.join(path, prefix + file))


def remove_prefix(prefix, path):
    """
    删除前缀
    """
    files = os.listdir(path)
    for file in files:
        os.rename(os.path.join(path, file), os.path.join(path, file.lstrip(prefix)))


if __name__ == '__main__':
	add_prefix(PREFIX_WINXP, WINXP_PATH)
	add_prefix(PREFIX_WIN7, WIN7_PATH)
	add_prefix(PREFIX_WIN8, WIN8_PATH)
	add_prefix(PREFIX_WIN10, WIN10_PATH)
