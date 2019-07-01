# encoding=utf-8
"""小工具函数"""

import os


def single(cls):
    """单例"""
    _instance = {}

    def warpper(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]

    return warpper


def to_abspath(path):
    """相对路径转绝对路径"""
    return os.path.abspath(path)


def list_to_sql_list(lists: list):
    """
    list转sql可用的list
    :param lists: list：[1,2,3,4,5]
    :return: '1','2''3','4','5'
    """

    lists = list(map(lambda x: f"''{x}''", lists))
    res = ','.join(lists)
    return res


def nan_cvt(x, dig: int = 4):
    """
    处理df数值，nan转为py中的None，小数默认保留小数点后4位
    :param x: 数值
    :param dig: 保留位数
    :return:
    """
    return None if str(x) == 'nan' else round(x, dig)


if __name__ == '__main__':
    path = 'dddd'

    print(to_abspath(path))