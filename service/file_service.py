# coding:utf-8
'''
Created on 2018/1/16.

@author: chk01
'''
import os
import requests
import random
import string
from tornado import gen


def createNoncestr(length=16, rule=string.ascii_letters + string.digits):
    return ''.join(random.sample(rule, length))


def download_url_img(url):
    fpath = 'userImg/download/' + createNoncestr() + '.png'

    fdir = fpath[:fpath.rfind('/')]
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    response = requests.get(url, cert=False)
    # img = Image.open(BytesIO(response.content)).convert("RGB")
    # img.save(fpath)

    with open(fpath, "wb") as code:
        code.write(response.content)

    # of = urllib.request.urlopen(url)
    #
    # # 下载文件
    # with open(fpath, "wb") as code:
    #     code.write(of.read())
    return fpath


def download_user_url_img(url):
    random_name = createNoncestr()
    fpath = 'userImg/download/{}/{}.png'.format(random_name, random_name)

    fdir = fpath[:fpath.rfind('/')]
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    response = requests.get(url, cert=False)
    # img = Image.open(BytesIO(response.content)).convert("RGB")
    # img.save(fpath)

    with open(fpath, "wb") as code:
        code.write(response.content)

    # of = urllib.request.urlopen(url)
    #
    # # 下载文件
    # with open(fpath, "wb") as code:
    #     code.write(of.read())
    return fpath
