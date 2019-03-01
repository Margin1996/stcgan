import cv2
import numpy as np
import random
import argparse
import logging
import glog as log
import os
import sys
from stcgan.shadow6.nets import *
import stcgan.shadow6.module as module
import glob
import mxnet as mx
# import pydevd


# pydevd.settrace('172.17.122.65', port=10203, stdoutToServer=True, stderrToServer=True)


def get_args(arglist=None):
    parser = argparse.ArgumentParser(
        description='Shadow Removel Params')
    parser.add_argument('-dbprefix', type=str, default='./ISTD_Dataset/train',
                        help='path of generated dataset prefix')
    parser.add_argument('-valprefix', type=str, default='./',
                        help='path of generated dataset prefix')
    parser.add_argument('-logfn', type=str,
                        default='deshadow_train', help='path to save log file')
    parser.add_argument('-gpuid', type=int, default=0,
                        help='gpu id, -1 for cpu')
    parser.add_argument('-lr', type=float, default=2e-3, help="learning rate")
    return parser.parse_args() if arglist is None else parser.parse_args(arglist)


def ferr(label, pred):
    pred = pred.ravel()
    label = label.ravel()
    return np.abs(label - (pred > 0.5)).sum() / label.shape[0]


if __name__ == '__main__':
    args = get_args()
    # environment setting
    log_file_name = args.logfn + '.log'
    log_file = open(log_file_name, 'w')
    log_file.close()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file_name)
    logger.addHandler(fh)

    if args.gpuid >= 0:
        context = mx.gpu(args.gpuid)
    else:
        context = mx.cpu()

    if not os.path.exists(args.dbprefix):
        logging.info(
            "training data not exist, pls check if the file path is correct.")
        sys.exit(0)
    if not os.path.exists("./result"):
        os.mkdir("./result")
    if not os.path.exists("./val_result"):
        os.mkdir("./val_result")
    if not os.path.exists("./trained_params"):
        os.mkdir("./trained_params")
    mstr = 'train'
    train_s_dir = os.path.join(args.dbprefix, '%s_A' % mstr)  # with shadow
    train_m_dir = os.path.join(args.dbprefix, '%s_B' % mstr)  # shadow mask
    train_g_dir = os.path.join(args.dbprefix, '%s_C' % mstr)  # gt
    val_s_dir = os.path.join(args.valprefix, 'test')
    # val_m_dir = os.path.join(args.valprefix, 'test_B')
    # val_g_dir = os.path.join(args.valprefix, 'test_C')
    assert os.path.exists(train_s_dir), '%s_A not exist!' % mstr
    assert os.path.exists(train_m_dir), '%s_B not exist!' % mstr
    assert os.path.exists(train_g_dir), '%s_C not exist!' % mstr
    filenms = os.listdir(train_s_dir)
    filenms_test = os.listdir(val_s_dir)
    # use rec file to load image.
    index = list(range(len(filenms)))
    index2 = list(range(len(filenms_test)))

    lr = args.lr
    beta1 = 0.5
    batch_size = 16
    # rand_shape = (batch_size, 100)
    num_epoch = 1000
    width = 256
    height = 256
    data_g1_shape = (batch_size, 3, width, height)
    data_g2_shape = (batch_size, 4, width, height)
    data_d1_shape = (batch_size, 4, width, height)
    data_d2_shape = (batch_size, 7, width, height)

    # initialize net
    gmod = module.GANModule(
        shadow_det_net_G1_v2(),
        shadow_removal_net_G2_v2(),
        shadow_det_net_D_v2(),
        bce_loss_v2(),
        l1_loss_v2(),
        context=context,
        data_g1_shape=data_g1_shape,
        data_g2_shape=data_g2_shape,
        data_d1_shape=data_d1_shape,
        data_d2_shape=data_d2_shape,
        hw=int(width / 32)
    )

    gmod.init_params(mx.init.Uniform(0.2))

    gmod.init_optimizer(lr)

    metric_acc1 = mx.metric.CustomMetric(ferr)
    metric_acc2 = mx.metric.CustomMetric(ferr)
    # load data
    for epoch in range(num_epoch):
        metric_acc1.reset()
        metric_acc2.reset()
        random.shuffle(index)
        random.shuffle(index2)
        data_s = np.zeros((batch_size, 3, width, height))
        data_m = np.zeros((batch_size, 1, width, height))
        data_g = np.zeros((batch_size, 3, width, height))

        for i in range(len(index) // batch_size):
            for j in range(batch_size):
                data_s_tmp = cv2.resize(cv2.imread(os.path.join(
                    train_s_dir, filenms[index[i * batch_size + j]])) / 255.0, (width, height))
                data_m_tmp = cv2.resize(cv2.imread(os.path.join(
                    train_m_dir, filenms[index[i * batch_size + j]]), cv2.IMREAD_GRAYSCALE) / 255.0, (width, height))
                data_m_tmp[data_m_tmp > 0.5] = 1.0
                data_m_tmp[data_m_tmp <= 0.5] = 0.0
                data_g_tmp = cv2.resize(cv2.imread(os.path.join(
                    train_g_dir, filenms[index[i * batch_size + j]])) / 255, (width, height))
                # random crop
                random_x = random.randint(0, data_s_tmp.shape[1] - height)
                random_y = random.randint(0, data_s_tmp.shape[0] - width)
                data_s[j, :, :, :] = np.transpose(
                    data_s_tmp[random_y: random_y + width, random_x: random_x + height, :], (2, 0, 1))
                data_m[j, 0, :, :] = data_m_tmp[random_y: random_y +
                                                width, random_x: random_x + height]
                data_g[j, :, :, :] = np.transpose(
                    data_g_tmp[random_y: random_y + width, random_x: random_x + height, :], (2, 0, 1))
            gmod.update(mx.nd.array(data_s, ctx=context), mx.nd.array(
                data_m, ctx=context), mx.nd.array(data_g, ctx=context))
            gmod.temp_label[:] = 0.0
            metric_acc1.update([gmod.temp_label], gmod.outputs_fake1)
            metric_acc2.update([gmod.temp_label], gmod.outputs_fake2)
            gmod.temp_label[:] = 1.0
            metric_acc1.update([gmod.temp_label], gmod.outputs_real1)
            metric_acc2.update([gmod.temp_label], gmod.outputs_real2)

            # training results
            log.info('epoch: %d, bce_loss is %.5f, adver_d1_loss is %.5f, l1_loss is %.5f, adver_d2_loss is %.5f'%(
                    epoch,gmod.loss[0, 0], gmod.loss[0, 1], gmod.loss[0, 2], gmod.loss[0, 3]))

        if epoch % 500 == 0 or epoch == num_epoch - 1:
            gmod.modG1.save_params('G1_epoch_{}.params'.format(epoch))
            gmod.modG2.save_params('G2_epoch_{}.params'.format(epoch))
            gmod.modD1.save_params('D1_epoch_{}.params'.format(epoch))
            gmod.modD2.save_params('D2_epoch_{}.params'.format(epoch))


        img_dir = glob.glob("test/*")
        img_name = []
        for i in img_dir:
            value = i[i.find("test/") + 5:]
            # print(value)
                # img_name.append(value)
                # dir_length = len(img_dir)
                # for i in range(dir_length):
            # img=cv2.imread(os.path.join(val_s_dir, i[i.find("test/") + 5:]))
            img_gt = cv2.imread(os.path.join(
                val_s_dir, i[i.find("test/") + 5:]))
            # w = cv2.imread(os.path.join(val_s_dir, i[i.find("test/") + 5:]))[1]
            # h = cv2.imread(os.path.join(val_s_dir, i[i.find("test/") + 5:]))[0]

            data_s_tmp = cv2.resize(cv2.imread(os.path.join(
                val_s_dir, i[i.find("test/") + 5:])) / 255.0, (width, height))
                # data_m_tmp = cv2.resize(cv2.imread(os.path.join(val_m_dir, filenms_test[index2[i]]),
                                                   # cv2.IMREAD_GRAYSCALE), (width, height))
                # data_g_tmp = cv2.resize(cv2.imread(os.path.join(
                    # val_g_dir, filenms_test[index2[i]])), (width, height))
                # random crop
            random_x = random.randint(0, data_s_tmp.shape[1] - height)
            random_y = random.randint(0, data_s_tmp.shape[0] - width)
            data_s[0, :, :, :] = np.transpose(
                data_s_tmp[random_y: random_y + width, random_x: random_x + height, :], (2, 0, 1))
                # data_m[0, 0, :, :] = data_m_tmp[random_y: random_y +
                                                # width, random_x: random_x + height]
                # data_g[0, :, :, :] = np.transpose(
                    # data_g_tmp[random_y: random_y + width, random_x: random_x + height, :], (2, 0, 1))
            gmod.forward(mx.nd.array(data_s, ctx=context))
                # cv2.imwrite('./val_result/sin_{}_{}.jpg'.format(epoch, i),
                            # np.round((np.transpose(data_s[0, :, :, :], (1, 2, 0))) * 255))
                # cv2.imwrite('./val_result/min_{}_{}.jpg'.format(epoch, i),
                            # data_m_tmp)
                # cv2.imwrite('./val_result/gin_{}_{}.jpg'.format(epoch, i),
                            # data_g_tmp)
                # cv2.imwrite('./SBU/shadow_free/'+img_name[i],#shadow free
                            # np.clip(np.round((np.transpose(gmod.temp_outG2.asnumpy()[0, :, :, :], (1, 2, 0)) + 1) / 2 * 255), 0, 255).astype(np.uint8))
                # cv2.imwrite('./SBU/shadow_mask/'+img_name[i],
                            # np.round((np.transpose(gmod.temp_outG1.asnumpy()[0, :, :, :], (1, 2, 0)) + 1) / 2 * 255))
                # shadow_remove

                # shadow_mask
            img = np.clip(np.round(np.transpose(gmod.temp_outG2.asnumpy()[
                          0, :, :, :], (1, 2, 0)) * 255), 0, 255).astype(np.uint8)
            img = cv2.resize(img, (img_gt.shape[1], img_gt.shape[0]))


            img2 = np.round((np.transpose(gmod.temp_outG1.asnumpy()[0, :, :, :], (1, 2, 0)) * 255).astype(np.uint8))
            img2 = cv2.resize(img2, (img_gt.shape[1], img_gt.shape[0]))

            cv2.imwrite('result/shadow_remove/' + value,
                        img)

            cv2.imwrite('result/shadow_mask/' + value,
                        img2)