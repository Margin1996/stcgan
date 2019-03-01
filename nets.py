import mxnet as mx
from mxnet import gluon
EPS =1e-6

def bce_loss():
    outG1 = mx.sym.Variable("outG1")
    outG1 = mx.sym.clip((outG1+1)/2.0,1e-6,0.999999)
    dbatch2 = mx.sym.Variable("dbatch2")
    # sigm = mx.sym.Activation(data=outG1, act_type='sigmoid')
    bceloss = -dbatch2 * mx.sym.log(outG1) + (1 - dbatch2) * mx.sym.log(1 - outG1)
    return bceloss

def l1_loss():
    outG2 = mx.sym.Variable("outG2")
    outG2 = mx.sym.clip((outG2+1)/2.0,1e-6,0.999999)
    dbatch3 = mx.sym.Variable("dbatch3")
    l1_loss = 100*mx.sym.abs(outG2-dbatch3)
    return l1_loss

def bce_loss_v2():
    outG1 = mx.sym.Variable("outG1")
    # outG1 = mx.sym.clip(outG1,1e-6,0.999999)
    dbatch2 = mx.sym.Variable("dbatch2")
    # dbatch2 = mx.sym.BlockGrad(dbatch2)
    # sigm = mx.sym.Activation(data=outG1, act_type='sigmoid')
    bceloss = mx.sym.LogisticRegressionOutput(data=outG1, label=dbatch2)
    # bceloss =-(dbatch2 * mx.sym.log(outG1+EPS) + (1 - dbatch2) * mx.sym.log(1 - outG1+EPS))
    return bceloss

def l1_loss_v2():
    outG2 = mx.sym.Variable("outG2")
    # outG2 = mx.sym.clip(outG2,1e-6,0.999999)
    dbatch3 = mx.sym.Variable("dbatch3")
    # l1_loss = mx.sym.square(outG2-dbatch3)
    # smooth_l1 = mx.sym.MakeLoss(mx.sym.smooth_l1(data=outG2-dbatch3, scalar=1, sigma=1))
    l1loss = mx.sym.MAERegressionOutput(data=outG2, label=dbatch3)
    return l1loss

def shadow_det_net_G1_v2():
    data = mx.sym.Variable("data")
    conv0 = mx.sym.Convolution(name ='conv0', data = data, num_filter=64, pad=(1,1),kernel=(3,3),stride=(2,2),no_bias=False,workspace = 1024)
    relu0 = mx.sym.LeakyReLU(data = conv0,act_type = 'leaky')

    conv1 = mx.sym.Convolution(name ='conv1', data = relu0, num_filter=128, pad=(1,1),kernel=(3,3),stride=(2,2),no_bias=False,workspace = 1024)
    bn1 = mx.sym.BatchNorm(data = conv1,use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu1 = mx.sym.LeakyReLU(data=bn1, act_type='leaky')

    conv2 = mx.sym.Convolution(name ='conv2', data = relu1, num_filter=256, pad=(1,1),kernel=(3,3),stride=(2,2),no_bias=False,workspace = 1024)
    bn2 = mx.sym.BatchNorm(data = conv2,use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu2 = mx.sym.LeakyReLU(data=bn2, act_type='leaky')

    conv3 = mx.sym.Convolution(name ='conv3', data = relu2, num_filter=512, pad=(1,1),kernel=(3,3),stride=(2,2),no_bias=False,workspace = 1024)
    bn3 = mx.sym.BatchNorm(data = conv3,use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu3 = mx.sym.LeakyReLU(data=bn3, act_type='leaky')

    conv4_1 = mx.sym.Convolution(name='conv4_1', data = relu3, num_filter=512, pad=(1,1),kernel=(3,3),stride=(2,2),no_bias=False,workspace = 1024)
    bn4_1 = mx.sym.BatchNorm(data=conv4_1,use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu4_1 = mx.sym.LeakyReLU(data=bn4_1, act_type='leaky')
    conv4_2 = mx.sym.Convolution(name='conv4_2', data = relu4_1, num_filter=512, pad=(1,1),kernel=(3,3),stride=(2,2),no_bias=False,workspace = 1024)
    bn4_2 = mx.sym.BatchNorm(data=conv4_2,use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu4_2 = mx.sym.LeakyReLU(data=bn4_2, act_type='leaky')
    conv4_3 = mx.sym.Convolution(name='conv4_3', data = relu4_2, num_filter=512, pad=(1,1),kernel=(3,3),stride=(2,2),no_bias=False,workspace = 1024)
    bn4_3 = mx.sym.BatchNorm(data=conv4_3,use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu4_3 = mx.sym.LeakyReLU(data=bn4_3, act_type='leaky')

    conv5 = mx.sym.Convolution(name='conv5', data = relu4_3, num_filter=512, pad=(1,1),kernel=(3,3),stride=(2,2),no_bias=False,workspace = 1024)
    bn5 = mx.sym.BatchNorm(data=conv5,use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu5 = mx.sym.LeakyReLU(data=bn5, act_type='leaky')

    conv6 = mx.sym.Deconvolution(name='conv6', data = relu5, num_filter=512, pad=(1,1),kernel=(4,4),stride=(2,2),no_bias=False,workspace = 1024)
    bn6 = mx.sym.BatchNorm(data=conv6,use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu6 = mx.sym.Activation(data=bn6, act_type='softrelu')

    data_7_1 = mx.sym.Concat(relu4_3,relu6,dim=1,name='data_7_1')
    conv7_1 = mx.sym.Deconvolution(name='conv7_1', data=data_7_1, num_filter=512, pad=(1, 1), kernel=(4, 4), stride=(2, 2),
                               no_bias=False, workspace=1024)
    bn7_1 = mx.sym.BatchNorm(data=conv7_1, use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu7_1 = mx.sym.Activation(data=bn7_1, act_type='softrelu')

    data_7_2 = mx.sym.Concat(relu4_2,relu7_1,dim=1,name='data_7_2')
    conv7_2 = mx.sym.Deconvolution(name='conv7_2', data=data_7_2, num_filter=512, pad=(1, 1), kernel=(4, 4), stride=(2, 2),
                               no_bias=False, workspace=1024)
    bn7_2 = mx.sym.BatchNorm(data=conv7_2, use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu7_2 = mx.sym.Activation(data=bn7_2, act_type='softrelu')

    data_7_3 = mx.sym.Concat(relu4_1, relu7_2, dim=1, name='data_7_3')
    conv7_3 = mx.sym.Deconvolution(name='conv7_3', data=data_7_3, num_filter=512, pad=(1, 1), kernel=(4, 4),
                                 stride=(2, 2),
                                 no_bias=False, workspace=1024)
    bn7_3 = mx.sym.BatchNorm(data=conv7_3, use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu7_3 = mx.sym.Activation(data=bn7_3, act_type='softrelu')

    data_8 = mx.sym.Concat(relu3, relu7_3, dim=1, name='data_8')
    conv8 = mx.sym.Deconvolution(name='conv8', data=data_8, num_filter=256, pad=(1, 1), kernel=(4, 4),
                                 stride=(2, 2),
                                 no_bias=False, workspace=1024)
    bn8 = mx.sym.BatchNorm(data=conv8, use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu8 = mx.sym.Activation(data=bn8, act_type='softrelu')

    data_9 = mx.sym.Concat(relu2, relu8, dim=1, name='data_9')
    conv9 = mx.sym.Deconvolution(name='conv9', data=data_9, num_filter=128, pad=(1, 1), kernel=(4, 4),
                                 stride=(2, 2),
                                 no_bias=False, workspace=1024)
    bn9 = mx.sym.BatchNorm(data=conv9, use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu9 = mx.sym.Activation(data=bn9, act_type='softrelu')

    data_10 = mx.sym.Concat(relu1, relu9, dim=1, name='data_10')
    conv10 = mx.sym.Deconvolution(name='conv_10', data=data_10, num_filter=64, pad=(1, 1), kernel=(4, 4),
                                 stride=(2, 2),
                                 no_bias=False, workspace=1024)
    bn10 = mx.sym.BatchNorm(data=conv10, use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu10 = mx.sym.Activation(data=bn10, act_type='softrelu')

    data_11 = mx.sym.Concat(relu0, relu10, dim=1, name='data_11')
    conv11 = mx.sym.Deconvolution(name='conv_11', data=data_11, num_filter=1, pad=(1, 1), kernel=(4, 4),
                                 stride=(2, 2),
                                 no_bias=False, workspace=1024)

    conv12 = mx.sym.Convolution(name='conv12', data=conv11, num_filter=1, pad=(1, 1), kernel=(3, 3), stride=(1, 1),
                               no_bias=False, workspace=1024)
    conv13 = mx.sym.Convolution(name='conv13', data=conv12, num_filter=1, pad=(1, 1), kernel=(3, 3), stride=(1, 1),
                               no_bias=False, workspace=1024)
    bn11 = mx.sym.BatchNorm(data=conv13, use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu11 = mx.sym.Activation(data=bn11, act_type='sigmoid')
    return relu11

def shadow_removal_net_G2_v2():
    data = mx.sym.Variable("data")
    conv0 = mx.sym.Convolution(name ='G2_conv0', data = data, num_filter=64, pad=(1,1),kernel=(3,3),stride=(2,2),no_bias=False,workspace = 1024)
    relu0 = mx.sym.LeakyReLU(data = conv0,act_type = 'leaky')

    conv1 = mx.sym.Convolution(name ='G2_conv1', data = relu0, num_filter=128, pad=(1,1),kernel=(3,3),stride=(2,2),no_bias=False,workspace = 1024)
    bn1 = mx.sym.BatchNorm(data = conv1,use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu1 = mx.sym.LeakyReLU(data=bn1, act_type='leaky')
    # pool1 = mx.sym.Pooling(name='G2_pool1', data=relu1 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='max')

    conv2 = mx.sym.Convolution(name ='G2_conv2', data = relu1, num_filter=256, pad=(1,1),kernel=(3,3),stride=(2,2),no_bias=False,workspace = 1024)
    bn2 = mx.sym.BatchNorm(data = conv2,use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu2 = mx.sym.LeakyReLU(data=bn2, act_type='leaky')
    # pool2 = mx.sym.Pooling(name='G2_pool2', data=relu2 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='max')

    conv3 = mx.sym.Convolution(name ='G2_conv3', data = relu2, num_filter=512, pad=(1,1),kernel=(3,3),stride=(2,2),no_bias=False,workspace = 1024)
    bn3 = mx.sym.BatchNorm(data = conv3,use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu3 = mx.sym.LeakyReLU(data=bn3, act_type='leaky')
    # pool3 = mx.sym.Pooling(name='G3_pool2', data=relu3 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='max')

    conv4_1 = mx.sym.Convolution(name='G2_conv4_1', data = relu3, num_filter=512, pad=(1,1),kernel=(3,3),stride=(2,2), no_bias=False,workspace = 1024)
    bn4_1 = mx.sym.BatchNorm(data=conv4_1,use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu4_1 = mx.sym.LeakyReLU(data=bn4_1, act_type='leaky')
    conv4_2 = mx.sym.Convolution(name='G2_conv4_2', data = relu4_1, num_filter=512, pad=(1,1),kernel=(3,3),stride=(2,2), no_bias=False,workspace = 1024)
    bn4_2 = mx.sym.BatchNorm(data=conv4_2,use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu4_2 = mx.sym.LeakyReLU(data=bn4_2, act_type='leaky')
    conv4_3 = mx.sym.Convolution(name='G2_conv4_3', data = relu4_2, num_filter=512, pad=(1,1),kernel=(3,3),stride=(2,2), no_bias=False,workspace = 1024)
    bn4_3 = mx.sym.BatchNorm(data=conv4_3,use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu4_3 = mx.sym.LeakyReLU(data=bn4_3, act_type='leaky')

    conv5 = mx.sym.Convolution(name='G2_conv5', data = relu4_3, num_filter=512, pad=(1,1),kernel=(3,3),stride=(2,2), no_bias=False,workspace = 1024)
    bn5 = mx.sym.BatchNorm(data=conv5,use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu5 = mx.sym.LeakyReLU(data=bn5, act_type='leaky')

    conv6 = mx.sym.Deconvolution(name='G2_conv6', data = relu5, num_filter=512, pad=(1,1),kernel=(4,4),stride=(2,2), no_bias=False,workspace = 1024)
    bn6 = mx.sym.BatchNorm(data=conv6,use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu6 = mx.sym.Activation(data=bn6, act_type='softrelu')

    data_7_1 = mx.sym.Concat(relu4_3, relu6, dim=1,name='G2_data_7_1')

    conv7_1 = mx.sym.Deconvolution(name='G2_conv7_1', data=data_7_1, num_filter=512, pad=(1, 1), kernel=(4, 4), stride=(2, 2), no_bias=False, workspace=1024)
    bn7_1 = mx.sym.BatchNorm(data=conv7_1, use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu7_1 = mx.sym.Activation(data=bn7_1, act_type='softrelu')

    data_7_2 = mx.sym.Concat(relu4_2, relu7_1,dim=1,name='G2_data_7_2')
    # data_7_2_up = mx.sym.UpSampling(data_7_2, name='data_7_2_up', scale=2, sample_type='nearest', workspace=1024)
    # conv7_2 = mx.sym.Convolution(name='G2_conv7_2', data=data_7_2_up, num_filter=512, pad=(1, 1), kernel=(3, 3),
    #                            stride=(1, 1), no_bias=False, workspace=1024)

    conv7_2 = mx.sym.Deconvolution(name='G2_conv7_2', data=data_7_2, num_filter=512, pad=(1, 1), kernel=(4, 4), stride=(2, 2),
                               no_bias=False, workspace=1024)
    bn7_2 = mx.sym.BatchNorm(data=conv7_2, use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu7_2 = mx.sym.Activation(data=bn7_2, act_type='softrelu')

    data_7_3 = mx.sym.Concat(relu4_1, relu7_2, dim=1, name='G2_data_7_3')
    # data_7_3_up = mx.sym.UpSampling(data_7_3, name='data_7_3_up', scale=2, sample_type='nearest', workspace=1024)
    # conv7_3 = mx.sym.Convolution(name='G2_conv7_3', data=data_7_3_up, num_filter=512, pad=(1, 1), kernel=(3, 3),
    #                            stride=(1, 1), no_bias=False, workspace=1024)
    conv7_3 = mx.sym.Deconvolution(name='G2_conv7_3', data=data_7_3, num_filter=512, pad=(1, 1), kernel=(4, 4),
                                 stride=(2, 2),
                                 no_bias=False, workspace=1024)
    bn7_3 = mx.sym.BatchNorm(data=conv7_3, use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu7_3 = mx.sym.Activation(data=bn7_3, act_type='softrelu')

    data_8 = mx.sym.Concat(relu3, relu7_3, dim=1, name='G2_data_8')
    # data_8_up = mx.sym.UpSampling(data_8,name='data_8_up',scale=2,sample_type ='nearest',workspace=1024)
    # conv8 =  mx.sym.Convolution(name='G2_conv8', data = data_8_up, num_filter=256, pad=(1,1),kernel=(3,3),stride=(1,1),no_bias=False,workspace = 1024)
    conv8 = mx.sym.Deconvolution(name='G2_conv8', data=data_8, num_filter=256, pad=(1, 1), kernel=(4, 4),
                                 stride=(2, 2),
                                 no_bias=False, workspace=1024)
    bn8 = mx.sym.BatchNorm(data=conv8, use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu8 = mx.sym.Activation(data=bn8, act_type='softrelu')

    data_9 = mx.sym.Concat(relu2, relu8, dim=1, name='G2_data_9')
    # data_9_up = mx.sym.UpSampling(data_9, name='data_9_up', scale=2, sample_type='nearest', workspace=1024)
    # conv9 = mx.sym.Convolution(name='G2_conv9', data=data_9_up, num_filter=128, pad=(1, 1), kernel=(3, 3),
    #                            stride=(1, 1), no_bias=False, workspace=1024)
    conv9 = mx.sym.Deconvolution(name='G2_conv9', data=data_9, num_filter=128, pad=(1, 1), kernel=(4, 4),
                                 stride=(2, 2),
                                 no_bias=False, workspace=1024)
    bn9 = mx.sym.BatchNorm(data=conv9, use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu9 = mx.sym.Activation(data=bn9, act_type='softrelu')

    # data_10_up = mx.sym.UpSampling(relu9, name='data_10_up', scale=2, sample_type='nearest', workspace=1024)
    data_10 = mx.sym.Concat(relu1, relu9, dim=1, name='G2_data_10')

    # data_10_up = mx.sym.UpSampling(data_10, name='data_10_up', scale=2, sample_type='nearest', workspace=1024)
    # conv10 = mx.sym.Convolution(name='G2_conv10', data=data_10_up, num_filter=64, pad=(1, 1), kernel=(3, 3),
    #                            stride=(1, 1), no_bias=False, workspace=1024)
    conv10 = mx.sym.Deconvolution(name='G2_conv10', data=data_10, num_filter=64, pad=(1, 1), kernel=(4, 4),
                                 stride=(2, 2),
                                 no_bias=False, workspace=1024)
    bn10 = mx.sym.BatchNorm(data=conv10, use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu10 = mx.sym.Activation(data=bn10, act_type='softrelu')

    data_11 = mx.sym.Concat(relu0, relu10, dim=1, name='G2_data_11')
    # data_11_up = mx.sym.UpSampling(data_11, name='data_11_up', scale=2, sample_type='nearest', workspace=1024)
    # conv11 = mx.sym.Convolution(name='G2_conv11', data=data_11_up, num_filter=3, pad=(1, 1), kernel=(3, 3),
    #                            stride=(1, 1), no_bias=False, workspace=1024)

    conv11 = mx.sym.Deconvolution(name='G2_conv11', data=data_11, num_filter=3, pad=(1, 1), kernel=(4, 4),
                                 stride=(2, 2),
                                 no_bias=False, workspace=1024)
    conv12 = mx.sym.Convolution(name='G2_conv12', data=conv11, num_filter=3, pad=(1, 1), kernel=(3, 3), stride=(1, 1),
                               no_bias=False, workspace=1024)
    conv13 = mx.sym.Convolution(name='G2_conv13', data=conv12, num_filter=3, pad=(1, 1), kernel=(3, 3), stride=(1, 1),
                               no_bias=False, workspace=1024)
    bn11 = mx.sym.BatchNorm(data=conv13, use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu11 = mx.sym.Activation(data=bn11, act_type='sigmoid')
    return relu11

def shadow_det_net_D_v2(str="D1"):
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')
    conv0 = mx.sym.Convolution(name ='%s_conv0'%str, data = data, num_filter=64, pad=(1,1),kernel=(4,4),stride=(2,2),no_bias=False,workspace = 1024)
    relu0 = mx.sym.LeakyReLU(data=conv0, act_type='leaky')

    conv1 = mx.sym.Convolution(name ='%s_conv1'%str, data = relu0, num_filter=128, pad=(1,1),kernel=(4,4),stride=(2,2),no_bias=False,workspace = 1024)
    bn1 = mx.sym.BatchNorm(data = conv1,use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu1 = mx.sym.LeakyReLU(data=bn1, act_type='leaky')

    conv2 = mx.sym.Convolution(name='%s_conv2'%str, data=relu1, num_filter=256, pad=(1, 1), kernel=(4, 4), stride=(2, 2),
                               no_bias=False, workspace=1024)
    bn2 = mx.sym.BatchNorm(data=conv2, use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu2 = mx.sym.LeakyReLU(data=bn2, act_type='leaky')

    conv3 = mx.sym.Convolution(name='%s_conv3'%str, data=relu2, num_filter=512, pad=(1, 1), kernel=(4, 4), stride=(2, 2),
                               no_bias=False, workspace=1024)
    bn3 = mx.sym.BatchNorm(data=conv3, use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu3 = mx.sym.LeakyReLU(data=bn3, act_type='leaky')

    conv4 = mx.sym.Convolution(name='%s_conv4'%str, data=relu3, num_filter=1, pad=(1, 1), kernel=(4, 4), stride=(2, 2),
                               no_bias=False, workspace=1024)
    bn4 = mx.sym.BatchNorm(data=conv4, use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu4 = mx.sym.Activation(data=bn4, act_type='sigmoid',name='%s_act4'%str)
    d5 = mx.sym.Flatten(relu4)
    #
    # fc = mx.sym.FullyConnected(d5, num_hidden=64, name="fc")
    # fc_loss = mx.sym.FullyConnected(data=d5, num_hidden=1, name="fc_dloss")
    # fc_out = mx.sym.Activation(data=fc_loss,act_type='sigmoid',name='fc_out')
    # dloss = mx.sym.MakeLoss(-(label * mx.sym.log(fc_loss+EPS) + (1 - label) * mx.sym.log(1 - fc_loss+EPS)))
    dloss = mx.sym.LogisticRegressionOutput(data=d5, label=label)
    # dloss = mx.sym.square(label-1)
    # dloss = 100*mx.sym.LogisticRegressionOutput(data = fc_loss, label = label, name='dloss')
    return dloss

def shadow_det_net_D2_v2():
    data = mx.sym.Variable('D2_data')
    label = mx.sym.Variable('D2_label')
    conv0 = mx.sym.Convolution(name ='D2_conv0', data = data, num_filter=64, pad=(1,1),kernel=(4,4),stride=(2,2),no_bias=False,workspace = 1024)
    relu0 = mx.sym.LeakyReLU(data=conv0, act_type='leaky')

    conv1 = mx.sym.Convolution(name ='D2_conv1', data = relu0, num_filter=128, pad=(1,1),kernel=(4,4),stride=(2,2),no_bias=False,workspace = 1024)
    bn1 = mx.sym.BatchNorm(data = conv1,use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu1 = mx.sym.LeakyReLU(data=bn1, act_type='leaky')

    conv2 = mx.sym.Convolution(name='D2_conv2', data=relu1, num_filter=256, pad=(1, 1), kernel=(4, 4), stride=(2, 2),
                               no_bias=False, workspace=1024)
    bn2 = mx.sym.BatchNorm(data=conv2, use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu2 = mx.sym.LeakyReLU(data=bn2, act_type='leaky')

    conv3 = mx.sym.Convolution(name='D2_conv3', data=relu2, num_filter=512, pad=(1, 1), kernel=(4, 4), stride=(2, 2),
                               no_bias=False, workspace=1024)
    bn3 = mx.sym.BatchNorm(data=conv3, use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu3 = mx.sym.LeakyReLU(data=bn3, act_type='leaky')

    conv4 = mx.sym.Convolution(name='D2_conv4', data=relu3, num_filter=1, pad=(1, 1), kernel=(4, 4), stride=(2, 2),
                               no_bias=False, workspace=1024)
    bn4 = mx.sym.BatchNorm(data=conv4, use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu4 = mx.sym.Activation(data=bn4, act_type='sigmoid')
    # d5 = mx.sym.Flatten(relu4)
    #
    # fc = mx.sym.FullyConnected(d5, num_hidden=64, name="fc")
    # fc_loss = mx.sym.FullyConnected(fc, num_hidden=1, name="fc_dloss")
    # dloss = 100*mx.sym.LogisticRegressionOutput(data = fc_loss, label = label, name='dloss')
    return relu4

def u_net_unit(inputsym, hasbn=False, re=None,):
    pass

def unet_sym():
    pass

def gnet(input_data, num_out, prefix):
    data = input_data
    conv0 = mx.sym.Convolution(name='%s_conv0'%prefix, data=data, num_filter=64, pad=(1, 1), kernel=(3, 3), stride=(2, 2),
                               no_bias=False, workspace=1024)
    relu0 = mx.sym.LeakyReLU(data=conv0, act_type='leaky')

    conv1 = mx.sym.Convolution(name='%s_conv1'%prefix, data=relu0, num_filter=128, pad=(1, 1), kernel=(3, 3), stride=(2, 2),
                               no_bias=False, workspace=1024)
    bn1 = mx.sym.BatchNorm(data=conv1, use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu1 = mx.sym.LeakyReLU(data=bn1, act_type='leaky')

    conv2 = mx.sym.Convolution(name='%s_conv2'%prefix, data=relu1, num_filter=256, pad=(1, 1), kernel=(3, 3), stride=(2, 2),
                               no_bias=False, workspace=1024)
    bn2 = mx.sym.BatchNorm(data=conv2, use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu2 = mx.sym.LeakyReLU(data=bn2, act_type='leaky')

    conv3 = mx.sym.Convolution(name='%s_conv3'%prefix, data=relu2, num_filter=512, pad=(1, 1), kernel=(3, 3), stride=(2, 2),
                               no_bias=False, workspace=1024)
    bn3 = mx.sym.BatchNorm(data=conv3, use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu3 = mx.sym.LeakyReLU(data=bn3, act_type='leaky')

    conv4_1 = mx.sym.Convolution(name='%s_conv4_1'%prefix, data=relu3, num_filter=512, pad=(1, 1), kernel=(3, 3), stride=(2, 2),
                                 no_bias=False, workspace=1024)
    bn4_1 = mx.sym.BatchNorm(data=conv4_1, use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu4_1 = mx.sym.LeakyReLU(data=bn4_1, act_type='leaky')
    conv4_2 = mx.sym.Convolution(name='%s_conv4_2'%prefix, data=relu4_1, num_filter=512, pad=(1, 1), kernel=(3, 3), stride=(2, 2),
                                 no_bias=False, workspace=1024)
    bn4_2 = mx.sym.BatchNorm(data=conv4_2, use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu4_2 = mx.sym.LeakyReLU(data=bn4_2, act_type='leaky')
    conv4_3 = mx.sym.Convolution(name='%s_conv4_3'%prefix, data=relu4_2, num_filter=512, pad=(1, 1), kernel=(3, 3), stride=(2, 2),
                                 no_bias=False, workspace=1024)
    bn4_3 = mx.sym.BatchNorm(data=conv4_3, use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu4_3 = mx.sym.LeakyReLU(data=bn4_3, act_type='leaky')

    conv5 = mx.sym.Convolution(name='%s_conv5'%prefix, data=relu4_3, num_filter=512, pad=(1, 1), kernel=(3, 3), stride=(2, 2),
                               no_bias=False, workspace=1024)
    bn5 = mx.sym.BatchNorm(data=conv5, use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu5 = mx.sym.LeakyReLU(data=bn5, act_type='leaky')

    conv6 = mx.sym.Deconvolution(name='%s_conv6'%prefix, data=relu5, num_filter=512, pad=(1, 1), kernel=(4, 4), stride=(2, 2),
                                 no_bias=False, workspace=1024)
    bn6 = mx.sym.BatchNorm(data=conv6, use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu6 = mx.sym.Activation(data=bn6, act_type='softrelu')

    data_7_1 = mx.sym.Concat(relu4_3, relu6, dim=1, name='%s_data_7_1'%prefix)
    conv7_1 = mx.sym.Deconvolution(name='%s_conv7_1'%prefix, data=data_7_1, num_filter=512, pad=(1, 1), kernel=(4, 4),
                                   stride=(2, 2),
                                   no_bias=False, workspace=1024)
    bn7_1 = mx.sym.BatchNorm(data=conv7_1, use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu7_1 = mx.sym.Activation(data=bn7_1, act_type='softrelu')

    data_7_2 = mx.sym.Concat(relu4_2, relu7_1, dim=1, name='%s_data_7_2'%prefix)
    conv7_2 = mx.sym.Deconvolution(name='%s_conv7_2'%prefix, data=data_7_2, num_filter=512, pad=(1, 1), kernel=(4, 4),
                                   stride=(2, 2),
                                   no_bias=False, workspace=1024)
    bn7_2 = mx.sym.BatchNorm(data=conv7_2, use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu7_2 = mx.sym.Activation(data=bn7_2, act_type='softrelu')

    data_7_3 = mx.sym.Concat(relu4_1, relu7_2, dim=1, name='data_7_3'%prefix)
    conv7_3 = mx.sym.Deconvolution(name='%s_conv7_3'%prefix, data=data_7_3, num_filter=512, pad=(1, 1), kernel=(4, 4),
                                   stride=(2, 2),
                                   no_bias=False, workspace=1024)
    bn7_3 = mx.sym.BatchNorm(data=conv7_3, use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu7_3 = mx.sym.Activation(data=bn7_3, act_type='softrelu')

    data_8 = mx.sym.Concat(relu3, relu7_3, dim=1, name='data_8'%prefix)
    conv8 = mx.sym.Deconvolution(name='%s_conv8'%prefix, data=data_8, num_filter=256, pad=(1, 1), kernel=(4, 4),
                                 stride=(2, 2),
                                 no_bias=False, workspace=1024)
    bn8 = mx.sym.BatchNorm(data=conv8, use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu8 = mx.sym.Activation(data=bn8, act_type='softrelu')

    data_9 = mx.sym.Concat(relu2, relu8, dim=1, name='%s_data_9'%prefix)
    conv9 = mx.sym.Deconvolution(name='%s_conv9'%prefix, data=data_9, num_filter=128, pad=(1, 1), kernel=(4, 4),
                                 stride=(2, 2),
                                 no_bias=False, workspace=1024)
    bn9 = mx.sym.BatchNorm(data=conv9, use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu9 = mx.sym.Activation(data=bn9, act_type='softrelu')

    data_10 = mx.sym.Concat(relu1, relu9, dim=1, name='%s_data_10'%prefix)
    conv10 = mx.sym.Deconvolution(name='%s_conv_10'%prefix, data=data_10, num_filter=64, pad=(1, 1), kernel=(4, 4),
                                  stride=(2, 2),
                                  no_bias=False, workspace=1024)
    bn10 = mx.sym.BatchNorm(data=conv10, use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu10 = mx.sym.Activation(data=bn10, act_type='softrelu')

    data_11 = mx.sym.Concat(relu0, relu10, dim=1, name='%s_data_11'%prefix)
    conv11 = mx.sym.Deconvolution(name='%s_conv_11'%prefix, data=data_11, num_filter=num_out, pad=(1, 1), kernel=(4, 4),
                                  stride=(2, 2),
                                  no_bias=False, workspace=1024)

    conv12 = mx.sym.Convolution(name='%s_conv12'%prefix, data=conv11, num_filter=1, pad=(1, 1), kernel=(3, 3), stride=(1, 1),
                                no_bias=False, workspace=1024)
    conv13 = mx.sym.Convolution(name='%s_conv13'%prefix, data=conv12, num_filter=1, pad=(1, 1), kernel=(3, 3), stride=(1, 1),
                                no_bias=False, workspace=1024)
    bn11 = mx.sym.BatchNorm(data=conv13, use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu11 = mx.sym.Activation(data=bn11, act_type='sigmoid')
    return relu11

def dnet(data, label, prefix):
    data = data
    label = label
    conv0 = mx.sym.Convolution(name ='%s_D2_conv0'%prefix, data = data, num_filter=64, pad=(1,1),kernel=(4,4),stride=(2,2),no_bias=False,workspace = 1024)
    relu0 = mx.sym.LeakyReLU(data=conv0, act_type='leaky')

    conv1 = mx.sym.Convolution(name ='%s_D2_conv1'%prefix, data = relu0, num_filter=128, pad=(1,1),kernel=(4,4),stride=(2,2),no_bias=False,workspace = 1024)
    bn1 = mx.sym.BatchNorm(name="%s_bn1"%prefix, data = conv1,use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu1 = mx.sym.LeakyReLU(data=bn1, act_type='leaky')

    conv2 = mx.sym.Convolution(name='%s_D2_conv2'%prefix, data=relu1, num_filter=256, pad=(1, 1), kernel=(4, 4), stride=(2, 2),
                               no_bias=False, workspace=1024)
    bn2 = mx.sym.BatchNorm(name="%s_bn2"%prefix,data=conv2, use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu2 = mx.sym.LeakyReLU(data=bn2, act_type='leaky')

    conv3 = mx.sym.Convolution(name='%s_D2_conv3'%prefix, data=relu2, num_filter=512, pad=(1, 1), kernel=(4, 4), stride=(2, 2),
                               no_bias=False, workspace=1024)
    bn3 = mx.sym.BatchNorm(name="%s_bn3"%prefix,data=conv3, use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu3 = mx.sym.LeakyReLU(data=bn3, act_type='leaky')

    conv4 = mx.sym.Convolution(name='%s_D2_conv4'%prefix, data=relu3, num_filter=1, pad=(1, 1), kernel=(4, 4), stride=(2, 2),
                               no_bias=False, workspace=1024)
    bn4 = mx.sym.BatchNorm(name="%s_bn4"%prefix, data=conv4, use_global_stats=False, fix_gamma=False, eps=0.000100)
    relu4 = mx.sym.Activation(data=bn4, act_type='sigmoid')

def end2end_sym(if_train=True):
    data = mx.sym.Variable("data")
    g1_out = gnet(data, num_out=1, prefix="g1")
    g2_input = mx.sym.Concat(data,g1_out, dim=1, name="g2_input")
    g2_out = gnet(g2_input, num_out=3, prefix="g2")
    label1 = mx.sym.Variable("d1_label")
    label2 = mx.sym.Variable("d2_label")
    d1_out = dnet(g1_out, label1, prefix="d1")
