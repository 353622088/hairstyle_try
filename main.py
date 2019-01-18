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
        user_img = "http://img.neuling.cn" + self.input("user_img")
        local_path = download_url_img(user_img)
        local_dict = get_landmark_dict(local_path)
        user_img_doc = {"userImgMat": dict(local_dict)}
        user_img_doc.update({"userImg": user_img, "userImgLocal": local_path})

        userImgId = mdb.user_img.insert(user_img_doc)
        fusionImg = fusion(local_path, local_dict)
        # os.remove(local_path)
        return self.finish(base.rtjson(fusionImg=fusionImg, userImgId=str(userImgId)))


class ChaneHairStyle(base.BaseHandler):
    def get(self):
        '''
        换一换
        :return:
        '''
        userImgId = self.input("userImgId")
        tempId = self.input("tempId", "temp1")
        userImgDoc = mdb.user_img.find_one({"_id": ObjectId(userImgId)})
        local_path = userImgDoc['userImgLocal']
        local_dict = userImgDoc['userImgMat']
        print(local_path)
        fusionImg = fusion(local_path, local_dict, tempId)
        return self.finish(base.rtjson(fusionImg=fusionImg))


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
    async def get(self):
        print('in')
        res = await self.doing()
        print(res)
        return self.finish('1')

    async def doing(self):
        print('do')
        # self.write('async')  # 返回消息
        dd = await gen.sleep(1)
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
