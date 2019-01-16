# coding:utf-8 
'''
created on 2019/1/15

@author:Dxq
'''
import tornado.ioloop
import tornado.web

import hashlib
import base64
import hmac
from common import base
import upyun

up = upyun.UpYun("qulifa", 'admin', 'jck20020808', timeout=120, endpoint=upyun.ED_AUTO)


class MainHandler(base.BaseHandler):
    def get(self):
        self.write("Welcome, Dxq")


class HairStyleTry(base.BaseHandler):
    def get(self):
        user_img = self.input("ok", "1")
        print(user_img)
        return self.finish({"user_img": [1, 2]})


class GetSignature(base.BaseHandler):
    def get(self):
        # py2+版本
        # signature = base64.b64encode(
        #     hmac.new(up.password, self.input('data'),
        #              digestmod=hashlib.sha1).digest()
        # ).decode()

        signature = base64.b64encode(
            bytes(hmac.new(bytes(up.password, "latin-1"), bytes(self.input('data'), "latin-1"),
                           digestmod=hashlib.sha1).hexdigest(), "latin-1")
        ).decode()

        return self.finish(base.rtjson(signature=signature, input=self.input('data'), upyun=up.password))


def make_app():
    return tornado.web.Application([
        (r"/", MainHandler), (r"/fusion", HairStyleTry), (r"/upyun/sign", GetSignature)
    ])


if __name__ == "__main__":
    app = make_app()
    app.listen(8080)
    tornado.ioloop.IOLoop.current().start()
