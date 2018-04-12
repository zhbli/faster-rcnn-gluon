from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models
from mxnet.contrib.ndarray import MultiBoxPrior
class RPN(gluon.HybridBlock):
    def __init__(self, num_classes, ctx,  **kwargs):
        super(RPN, self).__init__(**kwargs)
        # TODO: anchor size ?
        self.anchor_sizes = [.2, .272]
        self.anchor_ratios = [1, 2, .5]
        num_anchors = len(self.anchor_sizes) + len(self.anchor_ratios) - 1
        self.feature_extractor = self.get_pretrained_model(ctx)
        self.class_predictor = nn.Conv2D(num_anchors * (num_classes + 1), 3, padding=1)
        self.class_predictor.initialize(ctx=ctx)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.feature_extractor(x)
        anchors = MultiBoxPrior(x, sizes=self.anchor_sizes, ratios=self.anchor_ratios)
        class_pred = self.class_predictor(x)
        return anchors, class_pred

    def get_pretrained_model(self, ctx):
        # TODO: make sure end with which layer
        pretrained_net = models.vgg16(pretrained=True, ctx=ctx)
        feature_extractor = nn.HybridSequential()
        for layer in pretrained_net.features[:30]:
            feature_extractor.add(layer)
        return feature_extractor
