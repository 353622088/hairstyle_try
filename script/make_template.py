# coding:utf-8 
'''
created on 2019/1/10

@author:Dxq
'''
import os
import scipy.io as scio
import time
from PIL import Image
import functools
from common.utils import get_baseInfo_tx


def time_cal(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        t1 = time.time()
        r = func(*args, **kw)  # 先让函数运行一次,防止直接输出，将其赋值给一个变量
        if time.time() - t1 > 0.001:
            print('函数%s执行的时间为：%f' % (func.__name__, time.time() - t1))
        return r

    return wrapper


@time_cal
def get_landmark_dict(file_path):
    mat_file = file_path.split(".")[0] + '.mat'
    if os.path.exists(mat_file):
        landmark_dict = scio.loadmat(mat_file)
    else:
        landmark_dict = get_baseInfo_tx(file_path)
        # if landmark_dict['roll'] != 0:
        #     Image.open(file_path).rotate(-landmark_dict['roll']).save(file_path)
        #     landmark_dict = get_baseInfo_tx(file_path)
        scio.savemat(mat_file, landmark_dict)
    return landmark_dict


if __name__ == '__main__':
    '''
    将上传模板标准处理化
    '''
    root_dir = 'F:\project\dxq\hairstyle_try/resource/temp6'
    back_file = os.path.join(root_dir, 'ori.jpg')
    get_landmark_dict(back_file)
