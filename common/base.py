# coding:utf-8 
'''
created on 2019/1/15

@author:Dxq
'''
from tornado.web import RequestHandler
import time
import json


class BaseHandler(RequestHandler):
    def input(self, name, default=None, strip=True):
        return self._get_argument(name, default, self.request.arguments, strip)


def rtjson(code=1, **args):
    """return json"""
    if code == 1:
        args['status'] = 1
        args['response_time'] = int(time.time())
    else:
        args['status'] = 0
        args['error_code'] = code

    return json.dumps(args)
