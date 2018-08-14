#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 13:33:24 2017

@author: victoria
"""
from yllab import *
training = False
training = True
findBest = False
findBest = True
#sortKey = None
if training:
    pred('training.......')
    from train import *
    import trainInterface 
    trainInterface.train()
    
from val import *
import val

#import inferenceInterface 
#c.reload = inferenceInterface
import predictInterface 
c.predictInterface = predictInterface

#setMod('train')
evaluFun,sortKey = accEvalu,'acc'
if findBest:
    pred('auto Find Best Epoch .......')
    c.args.step = None
    
    epochs=sorted([findints(filename(p))[-1] for p in glob(c.weightsPrefix+'*')])
    df = autoFindBestEpoch(c,evaluFun,sortkey=sortKey, savefig='epoch.png',epochs=epochs)
#    row = df.loc[df[sortKey].argmax()]
    row = df.loc[df[sortKey].argmax()]
    restore =int(row.restore)
    df.to_csv('%s:%s_epoch:%s.csv'%(sortKey,row[sortKey],restore))
else:
    restore = -1
#%%
if __name__ == '__main__':
    pred('refine .......')
    c.args.restore = restore
    c.args.step = .2
#    reload(c.reload)
#    inference = inferenceInterface.inference 
    reload(predictInterface)
    inference = predictInterface.predict
#    c.inference = inference
    e = Evalu(evaluFun,
#              evaluName='restore-%s'%restore,
              valNames=c.names,
#              loadcsv=1,
#              logFormat='acc:{acc:.3f}, loss:{loss:.3f}',
              sortkey=sortKey,
#              loged=False,
              saveResoult=False,
              )
    c.names.sort(key=lambda x:readgt(x).shape[0])
    for name in c.names[:]:
        img,gt = readimg(name),readgt(name)
        prob = inference((name))
        re = prob.argmax(2)
        e.evalu(re,gt,name)
        
        gtc = labelToColor(gt,colors)
        rec = labelToColor(re,colors)
        smallGap = 10
#        show(img[::smallGap,::smallGap],gtc[::smallGap,::smallGap],(gt!=re)[::smallGap,::smallGap],rec[::smallGap,::smallGap])
#        diff = binaryDiff(re,gt)
#        show(img,diff,re)
#        show(img,diff)
#        show(diff)
#        yellowImg=gt[...,None]*img+(npa-[255,255,0]).astype(np.uint8)*~gt[...,None]
#        show(yellowImg,diff)
#        imsave(pathjoin(args.out,name+'.png'),uint8(re))
        imsave('val/%s.tif'%name,uint8(rec))
    print args.restore,e[sortKey].mean()



'''

'''