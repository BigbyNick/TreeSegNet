# coding: utf-8
'''
res-unet1-simg
取小图训练 

Parameters
----------
step : int
    填充黑边 将图片shape 调整为step的整数倍
'''
from yllab import *
from lib import *
import logging
logging.basicConfig(level=logging.INFO)
npm = lambda m:m.asnumpy()
npm = FunAddMagicMethod(npm)

import mxnet as mx
import random
from netdef import getNet

if __name__ == '__main__':
    from configManager import c  
    from train import args
    
else:
    from configManager import args,c
    
class SimpleBatch(object):
    def __init__(self, data, label, pad=0):
        self.data = data
        self.label = label
        self.pad = pad

    
labrgb = lambda lab:cv2.cvtColor(lab,cv2.COLOR_LAB2RGB)
randint = lambda x:np.random.randint(-x,x)
def imgAug(image,gt,prob=.5):
    if random.random() > prob:
        image = np.fliplr(image)
        gt = np.fliplr(gt)
    if random.random() > prob:
        image = np.flipud(image)
        gt = np.flipud(gt)
    return image,gt

def handleImgGt(imgs, gts,):
    for i in range(len(imgs)):
#        if np.random.randint(2):
#            imgs[i] = np.fliplr(imgs[i])
#            gts[i] = np.fliplr(gts[i])
#        if np.random.randint(2):
#            imgs[i] = np.flipud(imgs[i])
#            gts[i] = np.flipud(gts[i])
        imgs[i],gts[i] = imgAug(imgs[i],gts[i])
    if args.classn ==2:
        gts = gts >.5
    g.im=imgs;g.gt =gts
    imgs = imgs.transpose(0,3,1,2)/255.
    mximgs = map(mx.nd.array,[imgs])
    mxgtss = map(mx.nd.array,[gts])
    mxdata = SimpleBatch(mximgs,mxgtss)
    return mxdata

def readChannel(name, basenames=None):
#    kinds = ['_RGB.tif','_IRRG.tif','_lastools.jpg']
    kinds = ['_RGB.tif','_IRRG.tif','_dsm.tif']
    dirr = dirname(c['trainGlob'])
    if not basenames:
        basenames = kinds
    imgs = []
    if kinds[0] in basenames:
        path = pathjoin(dirr,name+kinds[0])
        img = imread(path)
        
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv = hsv.astype(np.int32)
        # adjust brightness
        hsv[:, :, 2] += random.randint(-15, 15)
        # adjust saturation
        hsv[:, :, 1] += random.randint(-10, 10)
        # adjust hue
        hsv[:, :, 0] += random.randint(-5, 5)
        hsv = np.clip(hsv, 0, 255)
        hsv = hsv.astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        imgs.append(img)
    if kinds[1] in basenames:
        path = pathjoin(dirr,name+kinds[1])
        img = imread(path)
        imgs.append(img[...,:1])
    if kinds[2] in basenames:
        path = pathjoin(dirr,name+kinds[2])
        img = imread(path)
        imgs.append(img[...,None])
    if len(imgs) == 1:
        return imgs[0]
    mimg = reduce(lambda x,y:np.append(x,y,2),imgs)
    return mimg
from collections import Iterator 
class GenSimg(Iterator):
    '''
    随机生成小图片simg及gt 的迭代器，默认使用1Gb内存作为图片缓存
    默认生成simg总面积≈所有图像总面积时 即结束
    '''
    def __init__(self, imggts, simgShape, handleImgGt=None,
                 batch=1, cache=None,iters=None,
                 timesPerRead=1,infinity=False):
        '''
        imggts: zip(jpgs,pngs)
        simgShape: simg的shape
        handleImgGt: 对输出结果运行handleImgGt(img,gt)处理后再返回
        batch: 每次返回的batch个数
        cache: 缓存图片数目, 默认缓存1Gb的数目
        timesPerRead: 平均每次读的图片使用多少次(不会影响总迭代次数),默认1次
        iters: 固定输出小图片的总数目，与batch无关
        infinity: 无限迭代
        '''
        if isinstance(simgShape,int):
            simgShape = (simgShape,simgShape)
        self.handleImgGt = handleImgGt
        self.imggts = imggts
        self.simgShape = simgShape
        self.batch = batch
        self._iters = iters
        self.iters = self._iters
        self.infinity = infinity
        
        hh,ww = simgShape
        jpg,png = imggts[0]
        img = readChannel(jpg)
        h,w = img.shape[:2]
        if cache is None:
            cache = max(1,int(5e9/img.nbytes))
        cache = min(cache,len(imggts))
        self.maxPerCache = int(cache*(h*w)*1./(hh*ww))* timesPerRead/batch
        self.cache = cache
        self.n = len(imggts)
        self._times = max(1,int(round(self.n*1./cache/timesPerRead)))
        self.times = self._times
        self.totaln = self.sn = iters or int((h*w)*self.n*1./(hh*ww))
        self.willn = iters or self.maxPerCache*self.times*batch
        self.count = 0
        self.reset()
        
        self.bytes = img.nbytes
        argsStr = '''imggts=%s pics in dir: %s, 
        simgShape=%s, 
        handleImgGt=%s,
        batch=%s, cache=%s,iters=%s,
        timesPerRead=%s, infinity=%s'''%(self.n , os.path.dirname(jpg) or './', simgShape, handleImgGt,
                                 batch, cache,iters,
                                 timesPerRead,infinity)
        generatorStr = '''maxPerCache=%s, readTimes=%s
        Will generator maxPerCache*readTimes*batch=%s'''%(self.maxPerCache, self.times,
                                                          self.willn)
        if iters:
            generatorStr = 'Will generator iters=%s'%iters
        self.__describe = '''GenSimg(%s)
        
        Total imgs Could generator %s simgs,
        %s simgs.
        '''%(argsStr,self.totaln,
             generatorStr,)
    def reset(self):
        if (self.times<=0 and self.iters is None) and not self.infinity:
            self.times = self._times
            raise StopIteration
        self.now = self.maxPerCache
        inds = np.random.choice(range(len(self.imggts)),self.cache,replace=False)
        datas = {}
        for ind in inds:
            jpg,png = self.imggts[ind]
            img,gt = readChannel(jpg),imread(png)
            datas[jpg] = img,gt
        self.data = self.datas = datas
        self.times -= 1
    def next(self):
        self.count += 1
        if (self.iters is not None) and not self.infinity:
            if self.iters <= 0:
                self.iters = self._iters
                raise StopIteration
            self.iters -= self.batch
        if self.now <= 0:
            self.reset()
        self.now -= 1
        hh,ww = self.simgShape
        datas = self.datas
        imgs, gts = [], []
        for t in range(self.batch):
            img,gt = datas[np.random.choice(datas.keys(),1,replace=False)[0]]
            h,w = img.shape[:2]
            i= np.random.randint(h-hh+1)
            j= np.random.randint(w-ww+1)
            (img,gt) =  img[i:i+hh,j:j+ww],gt[i:i+hh,j:j+ww]
            imgs.append(img), gts.append(gt)
        (imgs,gts) = map(np.array,(imgs,gts))
        if self.handleImgGt:
            return self.handleImgGt(imgs,gts)
        return (imgs,gts)
    @property
    def imgs(self):
        return [img for img,gt in self.datas.values()]
    @property
    def gts(self):
        return [gt for img,gt in self.datas.values()]
    def __str__(self):
        batch = self.batch
        n = len(self.datas)
        return self.__describe + \
        '''
    status:
        iter  in %s/%s(%.2f)
        batch in %s/%s(%.2f)
        cache imgs: %s
        cache size: %.2f MB
        '''%(self.count*batch,self.willn,self.count*1.*batch/self.willn,
            self.count,self._times*self.maxPerCache,
            self.count*1./(self._times*self.maxPerCache),
            n, (n*self.bytes/2**20))
        
    __repr__ = __str__
class GenSimgInMxnet(GenSimg):
    @property
    def provide_data(self):
        return [('data', (args.batch, 5, args.simgShape[0], args.simgShape[1]))]
    @property
    def provide_label(self):
        return  [('softmax1_label', (args.batch, args.simgShape[0], args.simgShape[1])),]


def saveNow(name = None):
    f=mx.callback.do_checkpoint(name or args.prefix)
    f(-1,mod.symbol,*mod.get_params())



    
default = dicto(
 gpu = 2,
 lr = 0.01,
 epochSize = 10000,
 step=64,
 window=64*2,
 classn=3
 )

for k in default.keys():
    if k not in args:
        args[k] = default[k]

args.names = zip(c.names,map(c.togt,c.names))

args.simgShape = args.window
if not isinstance(args.window,(tuple,list,np.ndarray)):
    args.simgShape = (args.window,args.window)

net = getNet(args.classn)

if args.resume:
    print('resume training from epoch {}'.format(args.resume))
    _, arg_params, aux_params = mx.model.load_checkpoint(
        args.prefix, args.resume)
else:
    arg_params = None
    aux_params = None

if 'plot' in args:
    mx.viz.plot_network(net, save_format='pdf', shape={
        'data': (1, 5, 640, 640),
        'softmax1_label': (1, 640, 640), }).render(args.prefix)
    exit(0)
mod = mx.mod.Module(
    symbol=net,
    context=[mx.gpu(k) for k in range(args.gpu)] if args.gpu!=1 else [mx.gpu(1)],
    data_names=('data',),
    label_names=('softmax1_label',)
)
c.mod = mod

#if 0:
gen = GenSimgInMxnet(args.names, args.simgShape, 
                      handleImgGt=handleImgGt,
                      batch=args.batch,
#                      cache=len(args.names),
                      iters=args.epochSize
                      )
#gen = GenSimgInMxnet(args.names,c.batch,handleImgGt=imgGtAdd0Fill(c.step))
g.gen = gen
total_steps = gen.totaln * args.epoch / gen.batch
lr_sch = mx.lr_scheduler.MultiFactorScheduler(
    step=[total_steps // 5 *1 ,total_steps // 5 *2 ,total_steps // 5 *3 ,total_steps // 5 * 4,int(total_steps / 5. * 4.5),], factor=0.1)
class Lrs(mx.lr_scheduler.MultiFactorScheduler):
    def __init__(self,*l,**kv):
        mx.lr_scheduler.MultiFactorScheduler.__init__(self,*l,**kv)
        self.num_update=None
    def __call__(self,num_update):
        lr = mx.lr_scheduler.MultiFactorScheduler.__call__(self,num_update)
        if self.num_update != num_update:
            stdout('\rstep:%s, lr:%s, '%(num_update, lr))
            self.num_update = num_update
        return lr
    
#lr_sch = lambda x:(log('\r %s, '%x) and 0.01)
#lr_sch = Lrs(
#    step=[total_steps // 5 *2 ,total_steps // 5 *3 ,total_steps // 5 * 4,int(total_steps / 5. * 4.5),], factor=0.1)
lr_sch = Lrs(
    step=[total_steps // 2, total_steps * 3// 4 , total_steps*15//16], factor=0.1)
    
def train():
    mod.fit(
        gen,
        begin_epoch=args.resume,
        arg_params=arg_params,
        aux_params=aux_params,
        batch_end_callback=mx.callback.Speedometer(args.batch),
        epoch_end_callback=mx.callback.do_checkpoint(args.prefix),
        optimizer='sgd',
        optimizer_params=(('learning_rate', args.lr), ('momentum', 0.9),
                          ('lr_scheduler', lr_sch), ('wd', 0.0005)),
        num_epoch=args.epoch)
if __name__ == '__main__':
    pass


if 0:
    #%%
    ne = g.gen.next()
#for ne in dd:
    ds,las = ne.data, ne.label
    d,la = npm-ds[0],npm-las[0]
    im = d.transpose(0,2,3,1)
    show(labrgb(uint8(im[0])));show(la)
