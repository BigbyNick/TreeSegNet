# coding: utf-8
import logging

logging.basicConfig(level=logging.INFO)

import mxnet as mx




def bottleneck(inputs, k):
    x = mx.sym.BatchNorm(data=inputs, momentum=0.99)
    x = mx.sym.Activation(data=x, act_type='relu')
    x = mx.sym.Convolution(data=x, kernel=(1,1), stride=(1,1), num_filter=k*4)
    x = mx.sym.Dropout(x, p=0.2)
    return x

def composite_function(inputs, dilate):
    x = mx.sym.BatchNorm(data=inputs, momentum=0.99)
    x = mx.sym.Activation(data=x, act_type='relu')
    x = mx.sym.Convolution(data=x, kernel=(3,3), stride=(1,1), pad=dilate, num_filter=k, dilate=dilate)
    x = mx.sym.Dropout(x, p=0.2)
    return x

def composite_function_bottleneck(inputs, dilate):
    x = bottleneck(inputs, k)
    x = composite_function(x, dilate)
    return x

def transition(inputs):
    x = mx.sym.BatchNorm(data=inputs, momentum=0.99)
    x = mx.sym.Convolution(data=x, kernel=(1,1), stride=(1,1), pad=(0,0), num_filter=k)
    # x = mx.sym.Dropout(x, p=0.2)
    return x

def dense_block(inputs, dilate):
    x1 = composite_function(inputs, dilate)
    x2 = composite_function(mx.sym.concat(inputs, x1, dim=1), dilate)
    x3 = composite_function(mx.sym.concat(inputs, x1, x2, dim=1), dilate)
    x4 = composite_function(mx.sym.concat(inputs, x1, x2, x3, dim=1), dilate)
    return mx.sym.concat(x1, x2, x3, x4, dim=1)

def conv(data, kernel=(3, 3), stride=(1, 1), pad=(0, 0), num_filter=None, name=None):
    return mx.sym.Convolution(data=data, kernel=kernel, stride=stride, pad=pad, num_filter=num_filter, name='conv_{}'.format(name))


def bn_relu(data, name):
    return mx.sym.Activation(data=mx.sym.BatchNorm(data=data, momentum=0.99, name='bn_{}'.format(name)), act_type='relu', name='relu_{}'.format(name))


def conv_bn_relu(data, kernel=(3, 3), stride=(1, 1), pad=(0, 0), num_filter=None, name=None):
    return bn_relu(conv(data, kernel, stride, pad, num_filter, 'conv_{}'.format(name)), 'relu_{}'.format(name))


def down_block(data, f, name):
    x = mx.sym.Pooling(data=data, kernel=(2,2), stride=(2,2), pool_type='max')
    # temp = conv_bn_relu(data, (3, 3), (2, 2), (1, 1),
    #                     f, 'layer1_{}'.format(name))
    x = conv_bn_relu(x, (3, 3), (1, 1), (1, 1),
                        f, 'layer2_{}'.format(name))
    bn = mx.sym.BatchNorm(data=conv(x, (3, 3), (1, 1), (1, 1), f, 'layer3_{}'.format(
        name)), momentum=0.99, name='layer3_bn_{}'.format(name))
    bn = bn + x
    act = mx.sym.Activation(data=bn, act_type='relu',
                            name='layer3_relu_{}'.format(name))
    return bn, act


def up_block(act, bn, f, p, name):
    x = mx.sym.UpSampling(
        act, num_filter=p, scale=2, sample_type='bilinear', name='upsample_{}'.format(name))
    # temp = mx.sym.Deconvolution(data=act, kernel=(3, 3), stride=(2, 2), pad=(
    #    1, 1), adj=(1, 1), num_filter=32, name='layer1_dconv_{}'.format(name))
    x = mx.sym.concat(bn, x, dim=1)
    x = conv_bn_relu(x, (1,1), (1,1), (0,0), f, 'layer_1x1_{}'.format(name))
    temp = conv_bn_relu(x, (3, 3), (1, 1), (1, 1),
                        f, 'layer2_{}'.format(name))
    bn = mx.sym.BatchNorm(data=conv(temp, (3, 3), (1, 1), (1, 1), f, 'layer3_{}'.format(
        name)), momentum=0.99, name='layer3_bn_{}'.format(name))
    bn = bn + x
    return mx.sym.Activation(data=bn, act_type='relu', name='layer3_relu_{}'.format(name))

k = 2
def getNet(n):
    global k
    k = n
    data = mx.sym.Variable('data')
    global rawData
    rawData = data
    x = conv_bn_relu(data, (3, 3), (1, 1), (1, 1), 64, 'conv0_1')
    net = conv_bn_relu(x, (3, 3), (1, 1), (1, 1), 64, 'conv0_2')
    bn1 = mx.sym.BatchNorm(data=conv(
        net, (3, 3), (1, 1), (1, 1), 64, 'conv0_3'), momentum=0.99, name='conv0_3_bn')
    bn1 = bn1 + x
    act1 = mx.sym.Activation(data=bn1, act_type='relu', name='conv0_3_relu')
    global ACT1
#    ACT1 = resnextBlock(act1,16,(1,1),False,getLayerName('short'),4)
    ACT1 = act1

    bn2, act2 = down_block(act1, 128, 'down1')
    bn3, act3 = down_block(act2, 256, 'down2')
    bn4, act4 = down_block(act3, 512, 'down3')
    bn5, act5 = down_block(act4, 512, 'down4')
    bn6, act6 = down_block(act5, 512, 'down5')

    bn7, act7 = down_block(act6, 512, 'down6')

    temp = up_block(act7, bn6, 512, 512, 'up6') 
    temp = up_block(temp, bn5, 512, 512, 'up5') 
    temp = up_block(temp, bn4, 256, 512, 'up4') 
    temp = up_block(temp, bn3, 128, 256, 'up3') 
    temp = up_block(temp, bn2, 64, 128, 'up2') 
    temp = up_block(temp, bn1, 32, 64, 'up1')
    score1 = conv(temp, (1, 1), (1, 1), (0, 0), 6, 'score1')
    net1 = mx.sym.SoftmaxOutput(score1, multi_output=True, name='softmax1')
    
    from yllab import load_data
    net1 = confusionTree(inputt=temp,tree=load_data('confusionTree'))
    return net1

__NAME_COUNT__ = {}
def getLayerName(name="None"):
    n = __NAME_COUNT__.get(name,0)
    n = n + 1
    __NAME_COUNT__[name] = n
    return name+'_%s'%n


def resnextBlock(data,num_filter, stride, dim_match, name,
                 num_group=32, bn_mom=0.9, workspace=256,):
    conv1 = mx.sym.Convolution(data=data, num_filter=int(num_filter*0.5), kernel=(1,1), stride=(1,1), pad=(0,0),
                                  no_bias=True, workspace=workspace, name=name + '_conv1')
    bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
    act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
    
    
    conv2 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*0.5), num_group=num_group, kernel=(3,3), stride=stride, pad=(1,1),
                                  no_bias=True, workspace=workspace, name=name + '_conv2')
    bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
    act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
    
    
    conv3 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                               workspace=workspace, name=name + '_conv3')
    bn3 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
    
    if dim_match:
        shortcut = data
    else:
        shortcut_conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                        workspace=workspace, name=name+'_sc')
        shortcut = mx.sym.BatchNorm(data=shortcut_conv, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_sc_bn')
    eltwise =  bn3 + shortcut
    return mx.sym.Activation(data=eltwise, act_type='relu', name=name + '_relu')

def pipe(data,filters):
    if filters == 1:
        out = conv_bn_relu(data, (3, 3), (1, 1), (1, 1),
                            1, getLayerName('conv_bn_relu'))
        return out
    fs = filters*4   #8--->16
    global ACT1
    data = mx.sym.concat(data,ACT1, dim=1)
    out = resnextBlock(data,fs,(1,1),False,getLayerName('resNext'),min(fs//4,32))
    return out 

def pipe2(inp,filters):
    if filters == 1:
        out = conv_bn_relu(inp, (3, 3), (1, 1), (1, 1),
                            1, getLayerName('conv_bn_relu'))
        return out
    layer1,layer2 = 5,3
    out1 = conv_bn_relu(inp, (3, 3), (1, 1), (1, 1),
                        filters*layer1, getLayerName('conv_bn_relu'))
    out = conv_bn_relu(out1, (3, 3), (1, 1), (1, 1),
                        filters*layer2, getLayerName('conv_bn_relu'))
#    out = conv(inp, (3, 3), (1, 1), (1, 1),
#                        filters, getLayerName('conv_bn_relu'))
    out = mx.sym.concat(inp,out, dim=1)
    return out 

def confusionTree(inputt=None,tree=None):
    if inputt is None:
        inputt = mx.sym.Variable('data')
    classn = sum(map(len,tree.keys()))
    probs = [0]*classn
    
    def walkTree(inp, tree, key):
        out = pipe(inp,len(key))
        if len(key) == 1:
            probs[key[0]]=out
        else:
            for k,v in tree.items():
                walkTree(out,v,k)
    walkTree(inputt,tree,tuple(range(classn)))
    out = mx.sym.concat(*probs, dim=1)
    net = mx.sym.SoftmaxOutput(out, multi_output=True, name='softmax1')
    return net

if __name__ == '__main__':
    pass
#    from yllab import *
    net = getNet(6)
#    net = confusionTree(tree=tre)
    mx.viz.plot_network(net, save_format='pdf', shape={
        'data': (1, 3, 640, 640),
        'softmax1_label': (1, 640, 640), }).render('TresegNet-short')