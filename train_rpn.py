import mxnet as mx
from RPN import RPN
from Dataset import getDataset
from mxnet import autograd, nd
num_cls = 20
batch_size = 64
train_data, test_data = getDataset()
ctx = mx.gpu(1)
net = RPN(num_cls, ctx) # TODO: num_class need to modify
for epoch in range(20):
    print('epoch: {}'.format(epoch))
    for i, batch in enumerate(train_data):
        print('batch')
        x = batch[0].as_in_context(ctx) # x.shape = (64, 3, 224, 224)
        y = batch[1].as_in_context(ctx) # y.shape = (64,)
        with autograd.record():
            anchors, class_pred_origin = net(x) # anchors.shape = (1, 784, 4), class_pred_origin.shape = (64, 84, 14, 14)
            # TODO: softmax ?
            # class_pred_origin.shape = (batch, anchor_num * (num_cls+1), height, width)
            class_pred_flatten = class_pred_origin.reshape(batch_size, -1) # class_pred_flatten.shape = (batch, anchor_num * (num_cls+1) * height * width)
            picked_anchors_index = class_pred_flatten.argmax(axis=1, keepdims=True) # shape = (batch,)
            temp1 = picked_anchors_index - picked_anchors_index % (num_cls+1)
            idx0 = nd.arange(batch_size, ctx=ctx).reshape(-1,1)
            idx1 = temp1 + nd.arange(num_cls+1, ctx=ctx)
            class_pred = class_pred_flatten[idx0, idx1]# class_pred.shape = (batch, num_cls + 1)
            nd.waitall()
            # class_target.shape = (batch, num_cls + 1)


