# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import lib
from lib import dicto,dirname, basename,os,log,fileJoinPath, pathjoin
from lib import show, loga, logl, imread, imsave

from configManager import (getImgGtNames, indexOf, readgt, readimg, 
                           setMod, togt, toimg)
from configManager import cf,c
# =============================================================================
# config BEGIN
# =============================================================================
cf.netdir = 'isprs'
cf.project = None
cf.experment = None

cf.trainGlob = u'/home/victoria/0-images/isprs/after/train/*_RGB.tif'
cf.toGtPath = lambda path:path.replace('_RGB.tif','_label.png')
cf.val = u'/home/victoria/0-images/isprs/after/val/*_RGB.tif'

cf.toValGtPath = None

cf.testGlob = u'/home/victoria/0-images/isprs/test/*_RGB.tif'
# =============================================================================
# config END
# =============================================================================


filePath = fileJoinPath(__file__)
jobDir = (os.path.split(dirname(filePath))[-1])
expDir = (os.path.split((filePath))[-1])

cf.project = cf.project or jobDir
cf.experment = cf.experment or expDir

cf.savename = '%s-%s-%s'%(cf.netdir,cf.experment,cf.project)

cf.toValGtPath = cf.toValGtPath or cf.toGtPath
#cf.valArgs = cf.valArgs or cf.trainArgs



c.update(cf)
c.cf = cf


c.weightsPrefix = fileJoinPath(__file__,pathjoin(c.tmpdir,'weights/%s-%s'%(c.netdir,c.experment)))
#show- map(readimg,c.names[:10])
if __name__ == '__main__':
    setMod('train')
    img = readimg(c.names[0])
    gt = readgt(c.names[0])
    show(img,gt)
    loga(gt)
    pass




