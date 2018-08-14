# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import sys,os
import numpy as np
import lib
from lib import dicto, glob, getArgvDic, findints,pathjoin
from lib import show, loga, logl, imread, imsave
from lib import Evalu,diceEvalu
from lib import *
from configManager import (getImgGtNames, indexOf, readgt, readimg, 
                           setMod, togt, toimg, makeValEnv, doc)
from train import c, cf, args

setMod('train')
setMod('test')

args.out = pathjoin(c.tmpdir,'val/tif-4-22')

# =============================================================================
# config BEGIN
# =============================================================================
args.update(
        restore=-1,
#        restore=34,
#        step=None,
        step=.2,
        )
# =============================================================================
# config END
# =============================================================================



if args.restore == -1:
    pas = [p[len(args.prefix):] for p in glob(args.prefix+'*')]
    args.restore = len(pas) and max(map(lambda s:len(findints(s)) and findints(s)[-1],pas))

makeValEnv(args)
accEvalu = lambda re,gt:{'acc':(re==gt).sum()*1./re.size,'loss':(~(re==gt)).sum()*1./re.size}

colors = np.array([(255, 255, 255),(0, 0, 255),(0, 255, 255),(0, 255, 0),
                   (255, 255, 0), (255, 0, 0),])/255.
    
import predictInterface 
c.predictInterface = predictInterface
if __name__ == '__main__':
    import predictInterface 
    c.predictInterface = predictInterface
    predict = predictInterface.predict 
#    c.predict = predict
    e = Evalu(accEvalu,
#              evaluName='restore-%s'%restore,
              valNames=c.names,
#              loadcsv=1,
              logFormat='acc:{acc:.3f}, loss:{loss:.3f}',
              sortkey='loss',
#              loged=False,
              saveResoult=False,
              )
#    c.names.sort(key=lambda x:readgt(x).shape[0])
    for name in c.names[::1]:
        img,gt = readimg(name),readgt
        prob = predict((name))
        re = prob.argmax(2)
#        e.evalu(re,gt,name)
        gt = re
        gtc = labelToColor(gt,colors)
        rec = labelToColor(re,colors)
        show(img[::10,::10],gtc[::10,::10],(gt!=re)[::10,::10],rec[::10,::10])
#        diff = binaryDiff(re,gt)
#        show(img,diff,re)
#        show(img,diff)
#        show(diff)
#        yellowImg=gt[...,None]*img+(npa-[255,255,0]).astype(np.uint8)*~gt[...,None]
#        show(yellowImg,diff)
#        imsave(pathjoin(args.out,name+'.tif'),uint8(rec))
        imsave(pathjoin(args.out,name+'.tif'),uint8(rec))
    
#    print args.restore,e.loss.mean()


#map(lambda n:show(readimg(n),e[n],readgt(n)),e.low(80).index[:])







