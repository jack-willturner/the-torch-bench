import torch
from models.ops import OP_REGISTRY
from models.workload_factory import resnet_configs, generate_random_config
from models.resnet import ResNet, BasicBlock, CIFARStem, ImageNetStem


def test_ops():

    for op in OP_REGISTRY:
        skeleton = resnet_configs["resnet18"]
        config = generate_random_config(
            skeleton.channels, skeleton.blocks, skeleton.strides, [op]
        )

        net = ResNet(
            BasicBlock, skeleton.blocks, config, num_classes=1000, stem=ImageNetStem
        )
        data = torch.rand((1, 3, 224, 224))

        y = net(data)

        assert y.size() == torch.Size((1, 1000))
