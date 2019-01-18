# coding:utf-8 
'''
created on 2019/1/18

@author:Dxq
'''
import numpy as np
import urllib.request
import cv2

url = 'http://img.neuling.cn/user/selfImg/023754_94aa33eac5cde22ce6cd8347557cfcf1.png'
resp = urllib.request.urlopen(url)
image = np.asarray(bytearray(resp.read()), dtype="uint8")
image = cv2.imdecode(image, cv2.IMREAD_COLOR)
cv2.imwrite("s.png", image)
