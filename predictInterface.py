# encoding: utf-8
'''
res-unet
全图预测 自动填充黑边 以适应上下采样

Parameters
----------
'''
from lib import *
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import mxnet as mx

from collections import namedtuple
from configManager import args,c

if __name__ == '__main__':
    import val
    
sym, arg_params, aux_params = mx.model.load_checkpoint(
    args.prefix, args.restore)
# print(sym.list_outputs())
Batch = namedtuple('Batch', ['data'])
#mod = mx.mod.Module(symbol=sym, label_names=None, context=mx.gpu())

args.simgShape = args.window
if not isinstance(args.window,(tuple,list,np.ndarray)):
    args.simgShape = (args.window,args.window)
mod = mx.mod.Module(symbol=sym, label_names=None, context=mx.gpu(1))
hh,ww = args.simgShape
mod.bind(for_training=False, data_shapes=[
         ('data', (1, 5, hh, ww))], label_shapes=None and [
                 ('softmax%d_label'%(i+1),(1,hh//2**i,ww//2**i)) for i in (0,)],
    force_rebind=False, )
mod.set_params(arg_params, aux_params, allow_missing=True)


def readChannel(name, basenames=None):
    kinds = ['_RGB.tif','_IRRG.tif','_dsm.tif']
    dirr = dirname(c['val'])
#    dirr = dirname(c['testGlob'])
    dirr = dirname(c.trainGlob)
#    dirr = dirname(c.testGlob)
    if not basenames:
        basenames = kinds
    imgs = []
    if kinds[0] in basenames:
        path = pathjoin(dirr,name+kinds[0])
        img = imread(path)
        shape = img.shape[:2]
        imgs.append(img)
    if kinds[1] in basenames:
        path = pathjoin(dirr,name+kinds[1])
        img = imread(path)
        imgs.append(img[...,:1])
    if kinds[2] in basenames:
        path = pathjoin(dirr,name+kinds[2])
        img = imread(path)
        if img.shape != shape:
            img = cv2.resize(img,shape)
        imgs.append(img[...,None])
    if len(imgs) == 1:
        return imgs[0]
    mimg = reduce(lambda x,y:np.append(x,y,2),imgs)
    return mimg

def predict(name):
    img = readChannel(name)/255.
#    img = img[::3,::3]
    def handleSimg(simg):
        simg = simg.transpose(2,0,1)
        mod.forward(Batch(data=[mx.nd.array(np.expand_dims(
                simg, 0))]), is_train=False)
        prob = mod.get_outputs()[0].asnumpy()[0]
        re= prob.transpose(1,2,0)
        return re

    re = autoSegmentWholeImg(img, args.simgShape, handleSimg, step=args.step, weightCore='gauss')
    return re       
    
if __name__ == '__main__':
    pass
    from test import *