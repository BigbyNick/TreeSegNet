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

setMod('val')
#setMod('test')
args.out = pathjoin(c.tmpdir,'val/png')

# =============================================================================
# config BEGIN
# =============================================================================
args.update(
#        restore=43,
        restore=28,
        step=None,
#        step=.2,
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
    c.names.sort(key=lambda x:readgt(x).shape[0])
    for name in c.names[:]:
        img,gt = readimg(name),readgt(name)
        prob = predict((name))
        re = prob.argmax(2)
        e.evalu(re,gt,name)
        
        gtc = labelToColor(gt,colors)
        rec = labelToColor(re,colors)
        show(img[::10,::10],gtc[::10,::10],(gt!=re)[::10,::10],rec[::10,::10])
#        diff = binaryDiff(re,gt)
#        show(img,diff,re)
#        show(img,diff)
#        show(diff)
#        yellowImg=gt[...,None]*img+(npa-[255,255,0]).astype(np.uint8)*~gt[...,None]
#        show(yellowImg,diff)
#        imsave(pathjoin(args.out,name+'.png'),uint8(re))
    
    print args.restore,e.loss.mean()


#map(lambda n:show(readimg(n),e[n],readgt(n)),e.low(80).index[:])






class ArgList(list):
    '''
    标记类 用于标记需要被autoFindBestParams函数迭代的参数列表
    '''
    pass

    
def autoFindBestParams(c, args,evaluFun,sortkey=None,savefig=False):
    '''遍历args里面 ArgList的所有参数组合 并通过sortkey 找出最佳参数组合
    
    Parameters
    ----------
    c : dicto
        即configManager 生成的测试集的所有环境配置 c
        包含args，数据配置，各类函数等
    args : dicto
        predict的参数，但需要包含 ArgList 类 将遍历ArgList的所有参数组合 并找出最佳参数组合
    evaluFun : Funcation
        用于评测的函数，用于Evalu类 需要返回dict对象
    sortkey : str, default None
        用于筛选时候的key 默认为df.columns[-1]
    
    Return: DataFrame
        每个参数组合及其评价的平均值
    '''
    iters = filter(lambda it:isinstance(it[1],ArgList),args.items())
    iters = sorted(iters,key=lambda x:len(x[1]),reverse=True)
    argsraw = args.copy()
    argsl = []
    args = dicto()
    
    k,vs = iters[0]
    lenn = len(iters)
    deep = 0
    tags = [0,]*lenn
    while deep>=0:
        vs = iters[deep][1]
        ind = tags[deep]
        if ind != len(vs):
            v = vs[ind]
            tags[deep]+=1
            key = iters[deep][0]
            args[key] = v
            if deep == lenn-1:
                argsl.append(args.copy())
            else:
                deep+=1
        else:
            tags[deep:]=[0]*(lenn-deep)
            deep -= 1
    assert len(argsl),"args don't have ArgList Values!!"
    pds,pddf = pd.Series, pd.DataFrame
    edic={}
    for arg in argsl:
        argsraw.update(arg)
        c.args.update(argsraw)
        e = Evalu(evaluFun,
                  evaluName='tmp',
                  sortkey=sortkey,
                  loged=False,
                  saveResoult=False,
                  )
        reload(c.predictInterface)
        predict = c.predictInterface.predict
        for name in c.names[::]:
            gt = c.readgt(name)
            prob = predict((name))
            re = prob.argmax(2)
#            from yllab import g
#            g.re,g.gt = re,gt
            e.evalu(re,gt,name)
    #        img = readimg(name)
    #        show(re,gt)
    #        show(img)
        if sortkey is None:
            sortkey = e.columns[-1]
        keys = tuple(arg.values())
        for k,v in arg.items():
            e[k] = v
        edic[keys] = e
        print 'arg: %s\n'%str(arg), e.mean()
    es = pddf(map(lambda x:pds(x.mean()), edic.values()))
    print '-'*20+'\nmax %s:\n'%sortkey,es.loc[es[sortkey].argmax()]
    print '\nmin %s:\n'%sortkey,es.loc[es[sortkey].argmin()]
    if len(iters) == 1:
        k = iters[0][0]
        import matplotlib.pyplot as plt
        df = es.copy()
        df = df.sort_values(k)
        plt.plot(df[k],df[sortkey],'--');plt.plot(df[k],df[sortkey],'rx')
        plt.xlabel(k);plt.ylabel(sortkey);plt.grid()
        if savefig:
            plt.savefig(savefig);
            plt.close()
        else:
            plt.show()    
    return es

def autoFindBestEpoch(c, evaluFun,sortkey=None,epochs=None,savefig=False):
    '''遍历所有epoch的weight  并通过测试集评估项sortkey 找出最佳epoch
    
    Parameters
    ----------
    c : dicto
        即configManager 生成的测试集的所有环境配置 c
        包含args，数据配置，各类函数等
    evaluFun : Funcation
        用于评测的函数，用于Evalu类 需要返回dict对象
    sortkey : str, default None
        用于筛选时候的key 默认为df.columns[-1]
    
    Return: DataFrame
        每个参数组合及其评价的平均值
    '''
    args = c.args
    if not isinstance(epochs,(tuple,list)) :
        pas = [p[len(args.prefix):] for p in glob(args.prefix+'*') if p[-4:]!='json']
        eps = map(lambda s:len(findints(s)) and findints(s)[-1],pas)
        maxx = len(eps) and max(eps)
        minn = len(eps) and min(eps)
        if isinstance(epochs,int):
            epochs = range(minn,maxx)[::epochs]+[maxx]
        else:
            epochs = range(minn,maxx+1)
    args['restore'] = ArgList(epochs)
#    print epochs
    df = autoFindBestParams(c, args, evaluFun,sortkey=sortkey,savefig=savefig)
    return df







