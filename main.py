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
from service.file_service import download_url_img
from service.fusion_service import fusion, get_landmark_dict
from config import mdb
from tornado import gen
import time

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

        user_img_doc = {"userImgMat": dict(user_img_dict)}
        user_img_doc.update({"userImg": user_img})
        user_img_id = mdb.user_img.insert(user_img_doc)
        t1 = time.time()
        print('get_base_info_waste::', t1 - t0)
        fusion_img = fusion(user_img, user_img_dict)
        print('all infer::', time.time() - t0)
        return self.finish(base.rtjson(fusionImg=fusion_img, userImgId=str(user_img_id)))


class ChaneHairStyle(base.BaseHandler):
    def get(self):
        '''
        换一换
        :return:
        '''
        t0 = time.time()
        user_img_id = self.input("userImgId")
        temp_id = self.input("tempId", "temp1")
        user_img_doc = mdb.user_img.find_one({"_id": ObjectId(user_img_id)})
        user_img = user_img_doc['userImg']
        user_img_dict = user_img_doc['userImgMat']
        fusion_img = fusion(user_img, user_img_dict, temp_id)
        print(time.time() - t0)
        return self.finish(base.rtjson(fusionImg=fusion_img))


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
