import mxnet as mx


def min_max_fun(arr):
    return mx.nd.clip(arr, 1e-9, 0.9999999)


def clip_grad(module):
    for grads in module._exec_group.grad_arrays:
        for grad in grads:
            tmp_grad = mx.nd.clip(grad, -0.1, 0.1)
            tmp_grad.copyto(grad)


class GANBaseModule(object):
    """Base class to hold gan data
    """

    def __init__(self,
                 symbol_generator1,
                 symbol_generator2,
                 context,
                 data_shape1,
                 data_shape2):
        # generator

        self.modG1 = mx.mod.Module(symbol=symbol_generator1,
                                   data_names=("data",),
                                   label_names=None,
                                   context=context)

        self.modG2 = mx.mod.Module(symbol=symbol_generator2,
                                   data_names=("data",),
                                   label_names=None,
                                   context=context)
        self.modG1.bind(data_shapes=[("data", data_shape1)])
        '''
        self.modG1.load_params('G1_epoch_1000.params')
        # '''
        self.modG2.bind(
            data_shapes=[("data", data_shape2)], inputs_need_grad=True)
        '''
        self.modG2.load_params('G2_epoch_1000.params')
        # '''


        # leave the discriminator
        self.temp_outG1 = None
        self.temp_outG2 = None

        self.temp_diffD1 = None
        self.temp_diffD2 = None

        self.temp_gradD1 = None
        self.temp_gradD2 = None
        self.context = context if isinstance(context, list) else [context]

        self.outputs_fake1 = None
        self.outputs_fake2 = None

        self.outputs_real1 = None
        self.outputs_real2 = None
        self.loss = mx.nd.zeros((1, 4), ctx=context).asnumpy()

        self.temp_rbatch1 = mx.io.DataBatch(
            [mx.nd.zeros(data_shape1, ctx=self.context[-1])], None)
        self.temp_rbatch2 = mx.io.DataBatch(
            [mx.nd.zeros(data_shape2, ctx=self.context[-1])], None)

    def _save_temp_gradD1(self):
        if self.temp_gradD1 is None:
            self.temp_gradD1 = [
                [grad.copyto(grad.context) for grad in grads]
                for grads in self.modD1._exec_group.grad_arrays]
        else:
            for gradsr, gradsf in zip(self.modD1._exec_group.grad_arrays, self.temp_gradD1):
                for gradr, gradf in zip(gradsr, gradsf):
                    gradr.copyto(gradf)

    def _save_temp_gradD2(self):
        if self.temp_gradD2 is None:
            self.temp_gradD2 = [
                [grad.copyto(grad.context) for grad in grads]
                for grads in self.modD2._exec_group.grad_arrays]
        else:
            for gradsr, gradsf in zip(self.modD2._exec_group.grad_arrays, self.temp_gradD2):
                for gradr, gradf in zip(gradsr, gradsf):
                    gradr.copyto(gradf)

    def _add_temp_gradD1(self):
        # add back saved gradient
        for gradsr, gradsf in zip(self.modD1._exec_group.grad_arrays, self.temp_gradD1):
            for gradr, gradf in zip(gradsr, gradsf):
                gradr += gradf

    def _add_temp_gradD2(self):
        # add back saved gradient
        for gradsr, gradsf in zip(self.modD2._exec_group.grad_arrays, self.temp_gradD2):
            for gradr, gradf in zip(gradsr, gradsf):
                gradr += gradf

    def init_params(self, *args, **kwargs):
        init = mx.init.Normal(0.1)
        self.modG1.init_params(init, **kwargs)
        self.modD1.init_params(init, **kwargs)
        self.modG2.init_params(init, **kwargs)
        self.modD2.init_params(init, **kwargs)
        self.bce_loss.init_params(init, **kwargs)
        self.l1_loss.init_params(init, **kwargs)

    def init_optimizer(self, lr):
        optimizer = "RMSProp"
        # self.modG1.init_optimizer(optimizer=optimizer, optimizer_params=(('learning_rate', lr),("wd", 0.0001),("lr_scheduler",mx.lr_scheduler.FactorScheduler(2500, factor=0.9, stop_factor_lr=1e-05))))
        # self.modD1.init_optimizer(optimizer=optimizer, optimizer_params=(('learning_rate', lr),("wd", 0.0001),("lr_scheduler",mx.lr_scheduler.FactorScheduler(2500, factor=0.9, stop_factor_lr=1e-05))))
        # self.modG2.init_optimizer(optimizer=optimizer, optimizer_params=(('learning_rate', lr),("wd", 0.0001),("lr_scheduler",mx.lr_scheduler.FactorScheduler(2500, factor=0.9, stop_factor_lr=1e-05))))
        # self.modD2.init_optimizer(optimizer=optimizer, optimizer_params=(('learning_rate', lr),("wd", 0.0001),("lr_scheduler",mx.lr_scheduler.FactorScheduler(2500, factor=0.9, stop_factor_lr=1e-05))))

        self.modG1.init_optimizer(optimizer=optimizer, optimizer_params=(
            ('clip_gradient', 0.1), ('learning_rate', 2 * lr), ("wd", 0.0001),
            ("lr_scheduler", mx.lr_scheduler.FactorScheduler(2500, factor=0.9, stop_factor_lr=1e-05))))
        self.modD1.init_optimizer(optimizer=optimizer, optimizer_params=(
            ('learning_rate', 4 * lr), ("wd", 0.0001),
            ("lr_scheduler", mx.lr_scheduler.FactorScheduler(2500, factor=0.9, stop_factor_lr=1e-05))))
        self.modG2.init_optimizer(optimizer=optimizer, optimizer_params=(
            ('clip_gradient', 0.1), ('learning_rate', lr), ("wd", 0.0001),
            ("lr_scheduler", mx.lr_scheduler.FactorScheduler(2500, factor=0.9, stop_factor_lr=1e-05))))
        self.modD2.init_optimizer(optimizer=optimizer, optimizer_params=(
            ('learning_rate', 2 * lr), ("wd", 0.0001),
            ("lr_scheduler", mx.lr_scheduler.FactorScheduler(2500, factor=0.9, stop_factor_lr=1e-05))))

        # self.bce_loss.init_optimizer(*args, **kwargs)
        # self.l1_loss.init_optimizer(*args, **kwargs)


class GANModule(GANBaseModule):
    """A thin wrapper of module to group generator and discriminator together in GAN.

    Example
    -------
    lr = 0.0005
    mod = GANModule(generator, encoder, context=mx.gpu()),
    mod.bind(data_shape=(3, 32, 32))
    mod.init_params(mx.init.Xavier())
    mod.init_optimizer("adam", optimizer_params={
        "learning_rate": lr,
    })

    for t, batch in enumerate(train_data):
        mod.update(batch)
        # update metrics
        mod.temp_label[:] = 0.0
        metricG.update_metric(mod.outputs_fake, [mod.temp_label])
        mod.temp_label[:] = 1.0
        metricD.update_metric(mod.outputs_real, [mod.temp_label])
        # visualize
        if t % 100 == 0:
            gen_image = mod.temp_outG[0].asnumpy()
            gen_diff = mod.temp_diffD[0].asnumpy()
            viz.imshow("gen_image", gen_image)
            viz.imshow("gen_diff", gen_diff)
    """

    def __init__(self,
                 symbol_generator1,
                 symbol_generator2,
                 symbol_encoder1,
                 bce_loss_generator,
                 l1_loss_generator,
                 context,
                 data_g1_shape,
                 data_g2_shape,
                 data_d1_shape,
                 data_d2_shape,
                 hw,
                 pos_label=0.9):
        super(GANModule, self).__init__(
            symbol_generator1, symbol_generator2, context, data_g1_shape, data_g2_shape)
        context = context if isinstance(context, list) else [context]
        self.batch_size = data_g1_shape[0]
        label_shape = (self.batch_size, 1, hw, hw)

        encoder1 = symbol_encoder1
        encoder2 = symbol_encoder1

        self.modD1 = mx.mod.Module(symbol=encoder1,
                                   data_names=("data",),
                                   label_names=("label",),
                                   context=context)

        self.modD1.bind(data_shapes=[("data", data_d1_shape)],
                        label_shapes=[("label", label_shape)],
                        inputs_need_grad=True)
        '''
        self.modD1.load_params('D1_epoch_1000.params')
        # '''
        self.modD2 = mx.mod.Module(symbol=encoder2,
                                   data_names=("data",),
                                   label_names=("label",),
                                   context=context)
        self.modD2.bind(data_shapes=[("data", data_d2_shape)],
                        label_shapes=[("label", label_shape)],
                        inputs_need_grad=True)
        '''
        self.modD2.load_params('D2_epoch_1000.params')
        # '''
        self.bce_loss = mx.mod.Module(symbol=bce_loss_generator,
                                      data_names=("outG1",),
                                      label_names=("dbatch2",),
                                      context=context)
        self.bce_loss.bind(data_shapes=[("outG1", (data_d1_shape[0], 1, data_d1_shape[2], data_d1_shape[3]))],
                           label_shapes=[
                               ("dbatch2", (data_d1_shape[0], 1, data_d1_shape[2], data_d1_shape[3]))],
                           inputs_need_grad=True)

        self.l1_loss = mx.mod.Module(symbol=l1_loss_generator,
                                     data_names=("outG2",),
                                     label_names=("dbatch3",),
                                     context=context)
        self.l1_loss.bind(data_shapes=[("outG2", (data_d2_shape[0], 3, data_d2_shape[2], data_d2_shape[3]))],
                          label_shapes=[
                              ("dbatch3", (data_d2_shape[0], 3, data_d2_shape[2], data_d2_shape[3]))],
                          inputs_need_grad=True)

        self.pos_label = pos_label
        self.temp_label = mx.nd.zeros(
            label_shape, ctx=context[-1])

    def update(self, dbatch1, dbatch2, dbatch3):
        """ shadow image, mask, shadow free image """
        # forward G1
        lam1 = 5
        lam2 = 0.1
        lam3 = 0.1
        # self.temp_rbatch1.data[0] = dbatch1
        self.modG1.forward(mx.io.DataBatch([dbatch1], [None]))
        outG1 = self.modG1.get_outputs()[0]

        self.bce_loss.forward(mx.io.DataBatch([outG1], [dbatch2]))
        bceloss = self.bce_loss.get_outputs()[0]
        # for_back = mx.nd.ones(self.bce_loss.get_outputs()[0].shape, self.context[-1])/self.batch_size
        clip_grad(self.bce_loss)
        self.bce_loss.backward()
        bce_loss_grad = self.bce_loss.get_input_grads()[0]
        self.loss[0, 0] = mx.nd.mean(bceloss).asnumpy()
        # self.modG1.backward([bce_loss_grad])
        D1_fake_input = mx.nd.zeros(
            (outG1.shape[0], 4, outG1.shape[2], outG1.shape[3]), self.context[-1])
        D1_real_input = mx.nd.zeros(
            (outG1.shape[0], 4, outG1.shape[2], outG1.shape[3]), self.context[-1])
        for i in range(outG1.shape[0]):
            D1_fake_input[i, :, :, :] = mx.nd.concat(
                dbatch1[i, :, :, :], outG1[i, :, :, :], dim=0)
            D1_real_input[i, :, :, :] = mx.nd.concat(
                dbatch1[i, :, :, :], dbatch2[i, :, :, :], dim=0)

        # forward D1
        self.temp_label[:] = 0
        self.modD1.forward(mx.io.DataBatch(
            [D1_fake_input], [self.temp_label]), is_train=True)
        loss_d1_1 = mx.nd.mean(self.modD1.get_outputs()[0]).asnumpy().copy()
        for_back = mx.nd.ones(self.modD1.get_outputs()[
                              0].shape, self.context[-1]) / self.batch_size * lam1
        clip_grad(self.modD1)
        self.modD1.backward([for_back])
        self._save_temp_gradD1()

        self.temp_label[:] = 1
        self.modD1.forward(mx.io.DataBatch(
            [D1_fake_input], [self.temp_label]), is_train=True)
        for_back = mx.nd.ones(self.modD1.get_outputs()[
                              0].shape, self.context[-1]) / self.batch_size * lam1
        clip_grad(self.modD1)
        self.modD1.backward([for_back])
        diffD1 = self.modD1.get_input_grads()[0].copy()

        self.outputs_fake1 = [x.copyto(x.context)
                              for x in self.modD1.get_outputs()]

        self.temp_label[:] = 1
        self.modD1.forward(mx.io.DataBatch(
            [D1_real_input], [self.temp_label]), is_train=True)
        # part2 = mx.nd.log(min_max_fun(self.modD1.get_outputs()[0]))
        # self.loss[0,1] = mx.nd.mean(0.1*(part1+part2)).asnumpy()
        loss_d1_2 = mx.nd.mean(self.modD1.get_outputs()[0]).asnumpy().copy()
        self.loss[0, 1] = loss_d1_1 + loss_d1_2

        for_back = mx.nd.ones(self.modD1.get_outputs()[
                              0].shape, self.context[-1]) / self.batch_size * lam1
        clip_grad(self.modD1)
        self.modD1.backward([for_back])
        self._add_temp_gradD1()
        self.outputs_real1 = self.modD1.get_outputs()

        # forward G2
        self.temp_rbatch2.data[0] = D1_fake_input
        self.modG2.forward(self.temp_rbatch2)
        outG2 = self.modG2.get_outputs()[0]

        self.l1_loss.forward(mx.io.DataBatch([outG2], [dbatch3]))  #compute the L1_loss between G2_out and free_shadow_img
        l1loss = self.l1_loss.get_outputs()[0]
        for_back = mx.nd.ones(self.l1_loss.get_outputs()[
                              0].shape, self.context[-1]) / self.batch_size * lam2
        clip_grad(self.l1_loss)
        self.l1_loss.backward([for_back])
        l1_loss_grad = self.l1_loss.get_input_grads()[0]
        self.loss[0, 2] = mx.nd.mean(l1loss).asnumpy()

        D2_fake_input = mx.nd.zeros(
            (outG2.shape[0], 7, outG2.shape[2], outG2.shape[3]), self.context[-1])
        D2_real_input = mx.nd.zeros(
            (outG2.shape[0], 7, outG2.shape[2], outG2.shape[3]), self.context[-1])
        for i in range(outG2.shape[0]):
            D2_fake_input[i, :, :, :] = mx.nd.concat(
                D1_fake_input[i, :, :, :], outG2[i, :, :, :], dim=0)
            D2_real_input[i, :, :, :] = mx.nd.concat(
                D1_real_input[i, :, :, :], dbatch3[i, :, :, :], dim=0)

        # forward D2
        self.temp_label[:] = 0
        self.modD2.forward(mx.io.DataBatch(
            [D2_fake_input], [self.temp_label]), is_train=True)
        # part1 = mx.nd.log(1 - min_max_fun(self.modD2.get_outputs()[0]))
        loss_d2_1 = mx.nd.mean(self.modD2.get_outputs()[0]).asnumpy().copy()
        # self.loss[0, 3] = self.loss[0, 3] + mx.nd.mean(self.modD2.get_outputs()[0]).asnumpy()
        for_back = mx.nd.ones(self.modD2.get_outputs()[
                              0].shape, self.context[-1]) / self.batch_size * lam3
        clip_grad(self.modD2)
        self.modD2.backward([for_back])
        self._save_temp_gradD2()

        self.temp_label[:] = 1
        self.modD2.forward(mx.io.DataBatch(
            [D2_fake_input], [self.temp_label]), is_train=True)
        self.loss[0, 3] = self.loss[0, 3] + \
            mx.nd.mean(mx.nd.mean(self.modD2.get_outputs()[0])).asnumpy()
        for_back = mx.nd.ones(self.modD2.get_outputs()[
                              0].shape, self.context[-1]) / self.batch_size * lam3
        clip_grad(self.modD2)
        self.modD2.backward([for_back])
        diffD2 = self.modD2.get_input_grads()[0].copy()
        self.outputs_fake2 = [x.copyto(x.context)
                              for x in self.modD2.get_outputs()]

        # for updating G2
        self.temp_label[:] = 1
        self.modD2.forward(mx.io.DataBatch(
            [D2_real_input], [self.temp_label]), is_train=True)
        loss_d2_2 = mx.nd.mean(self.modD2.get_outputs()[0]).asnumpy().copy()
        self.loss[0, 3] = loss_d2_1 + loss_d2_2
        for_back = mx.nd.ones(self.modD2.get_outputs()[
                              0].shape, self.context[-1]) / self.batch_size * lam3
        clip_grad(self.modD2)
        self.modD2.backward([for_back])
        self._add_temp_gradD2()
        # diffD2 = self.modD2.get_input_grads()[0]
        self.outputs_real2 = self.modD2.get_outputs()

        # self.outputs_fake2 = [x.copyto(x.context) for x in self.modD2.get_outputs()]

        # update D2
        # self.temp_label[:] = self.pos_label
        # self.modD2.forward(mx.io.DataBatch([D2_real_input], [self.temp_label]), is_train=True)
        # self.modD2.backward()
        # self._add_temp_gradD2()
        self.modD2.update()
        # self.outputs_real2 = self.modD2.get_outputs()
        self.temp_outG2 = outG2
        # self.temp_diffD2 = diffD2

        # update D1
        # self.temp_label[:] = self.pos_label
        # self.modD1.forward(mx.io.DataBatch([D1_real_input], [self.temp_label]), is_train=True)
        # self.modD1.backward()
        # self._add_temp_gradD1()
        self.modD1.update()
        # self.outputs_real1 = self.modD1.get_outputs()
        self.temp_outG1 = outG1
        # self.temp_diffD1 = diffD1

        # update G2
        # self.modG2.backward([diffD2[:,4:,:,:]])
        clip_grad(self.modG2)
        self.modG2.backward([diffD2[:, 4:, :, :] + l1_loss_grad])
        tmp_G2_grads = self.modG2.get_input_grads()[0]
        self.modG2.update()

        # update G1
        # self.modG1.backward([mx.nd.slice_axis(diffD1,axis=1,begin=3,end=4)
        #                      + mx.nd.slice_axis(diffD2,axis=1,begin=3,end=4)])
        clip_grad(self.modG1)
        self.modG1.backward([mx.nd.slice_axis(diffD1, axis=1, begin=3, end=4)
                             + bce_loss_grad +
                             mx.nd.slice_axis(diffD2, axis=1, begin=3, end=4)
                             + mx.nd.slice_axis(tmp_G2_grads, axis=1, begin=3, end=4)])
        # self.modG1.backward([bce_loss_grad])
        self.modG1.update()

    def forward(self, dbatch1):
        self.temp_rbatch1.data[0] = dbatch1
        self.modG1.forward(self.temp_rbatch1)
        outG1 = self.modG1.get_outputs()[0]
        D1_fake_input = mx.nd.zeros(
            (outG1.shape[0], 4, outG1.shape[2], outG1.shape[3]), self.context[-1])
        for i in range(outG1.shape[0]):
            D1_fake_input[i, :, :, :] = mx.nd.concat(
                dbatch1[i, :, :, :], outG1[i, :, :, :], dim=0)
        # forward G2
        self.temp_rbatch2.data[0] = D1_fake_input
        self.modG2.forward(self.temp_rbatch2)
        outG2 = self.modG2.get_outputs()[0]
        self.temp_outG2 = outG2
        self.temp_outG1 = outG1
