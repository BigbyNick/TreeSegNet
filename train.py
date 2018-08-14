# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from lib import *
import sys,os
import lib
from lib import dicto, glob, getArgvDic,filename
from lib import show, loga, logl, imread, imsave
from configManager import (getImgGtNames, indexOf, readgt, readimg, 
                           setMod, togt, toimg, makeTrainEnv)

from config import c, cf

setMod('train')

from configManager import args
args.names = getImgGtNames(c.names)[:]
args.prefix = c.weightsPrefix
args.classn = 6
#args.window = (64*10,64*10)
args.window = (64*8,64*8)
#args.window = (64*1,64*1)
[ 20.     ,  29.96875]
# =============================================================================
# config BEGIN
# =============================================================================
args.update(
#        batch=8,
#        batch=1,  
        batch=2,  #4G*2
#        batch=4, #8G*2
#        epoch=50,
        epoch=80,
        resume=0,
        epochSize = 10000,
        )
# =============================================================================
# config END
# =============================================================================




argListt, argsFromSys = getArgvDic()
args.update(argsFromSys)

makeTrainEnv(args)
c.args=(args)
if __name__ == '__main__':
    import trainInterface as train
    train.train()
    pass

