# -*- coding: utf-8 -*-
import sys
from os.path import abspath,join
# 填写 deep517/lib/ 的绝对路径
absLibpPath = None

if absLibpPath is None:
    _path = (abspath(__file__))
    absLibpPath = absLibpPath or join(_path[:_path.index('deep517')+7],'lib')
    
if absLibpPath not in sys.path:
    sys.path = [absLibpPath]+sys.path

from yllibInterface import *
import configManager

if __name__ == '__main__':
    print(absLibpPath)
    pass
