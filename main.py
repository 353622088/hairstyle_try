# coding:utf-8 
'''
created on 2019/1/15

@author:Dxq
'''
import tornado.ioloop
import tornado.web
from bson import ObjectId
import hashlib
import base64
import hmac
from common import base
import upyun
from service.file_service import download_user_url_img
from service.fusion_service import fusion, get_landmark_dict
from config import mdb
from tornado import gen
import time
import random
import os

up = upyun.UpYun("qulifa", 'admin', 'jck20020808', timeout=120, endpoint=upyun.ED_AUTO)


class MainHandler(base.BaseHandler):
    def get(self):
        self.write("Welcome, Dxq")


class HairStyleTry(base.BaseHandler):
    '''
    功能：上传自拍照试戴
    :return:返回服务器本地图片地址（需要策略定时删除）
    '''

    def post(self):
        t0 = time.time()
        # input img and img_dict
        user_img = "http://img.neuling.cn" + self.input("user_img")
        user_img_dict = get_landmark_dict(user_img, 'url')
        user_local_img = download_user_url_img(user_img)
        user_img_doc = {"userImgMat": dict(user_img_dict)}
        user_img_doc.update({"userImg": user_img, "userLocalImg": user_local_img})

        t1 = time.time()
        print('get_base_info_waste::', t1 - t0)
        temp_list = ['temp1', 'temp2', 'temp3', 'temp4', 'temp5']
        temp_id = random.sample(temp_list, 1)[0]
        user_img_doc.update({"tempIds": [temp_id]})

        user_img_id = mdb.user_img.insert(user_img_doc)
        _, fusion_img = fusion(user_local_img, user_img_dict, temp_id)
        print('all infer::', time.time() - t0)
        return self.finish(base.rtjson(fusionImg=fusion_img, userImgId=str(user_img_id), tempId=temp_id))


class ChaneHairStyle(base.BaseHandler):
    def get(self):
        '''
        换一换
        :return:
        '''
        t0 = time.time()
        user_img_id = self.input("userImgId")

        user_img_doc = mdb.user_img.find_one({"_id": ObjectId(user_img_id)})
        user_local_img = user_img_doc['userLocalImg']
        user_img_dict = user_img_doc['userImgMat']
        temp_old_ids = user_img_doc['tempIds']

        # 一对一换发型
        temp_id = self.input("tempId", "")

        if not temp_id:
            # 随机换一换
            temp_list = ['temp1', 'temp2', 'temp3', 'temp4', 'temp5']
            index = len(temp_old_ids) % len(temp_list)
            temp_list_del = list(set(temp_list) - set(temp_old_ids))
            temp_id = random.sample(temp_list_del, 1)[0] if len(temp_list_del) > 0 else temp_old_ids[index]

        file_name = os.path.basename(user_local_img).split('.')[0]
        fusion_img = "userImg/download/{}/{}_thum.png".format(file_name, temp_id)
        t1 = time.time()
        print('before::', t1 - t0)
        if not os.path.exists(fusion_img):
            # _, fusion_img = fusion(user_local_img, user_img_dict, temp_id)
            fusion_img, _ = fusion(user_local_img, user_img_dict, temp_id)
        temp_old_ids.append(temp_id)

        mdb.user_img.update_one({"_id": ObjectId(user_img_id)}, {"$set": {"tempIds": temp_old_ids}})
        print('fusion::', time.time() - t1)
        return self.finish(base.rtjson(fusionImg=fusion_img, tempId=temp_id))


class GetSignature(base.BaseHandler):
    def get(self):
        # py2+版本
        # signature = base64.b64encode(
        #     hmac.new(up.password, self.input('data'),
        #              digestmod=hashlib.sha1).digest()
        # ).decode()

        signature = base64.b64encode(
            hmac.new(bytes(up.password, "utf-8"), bytes(self.input('data'), "utf-8"),
                     digestmod=hashlib.sha1).digest()).decode()

        return self.finish(base.rtjson(signature=signature, input=self.input('data'), upyun=up.password))


class Test(base.BaseHandler):
    def get(self):
        print('in')
        res = self.doing()
        print(res)
        return self.finish('1')

    def doing(self):
        print('do')
        # self.write('async')  # 返回消息
        dd = gen.sleep(1)
        print('do2')
        # raise gen.Return(2)
        return dd


def make_app():
    return tornado.web.Application([
        (r"/", MainHandler), (r"/fusion", HairStyleTry), (r"/upyun/sign", GetSignature),
        (r"/test", Test), (r"/change", ChaneHairStyle),
    ])


if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    print('8888')
    tornado.ioloop.IOLoop.current().start()
