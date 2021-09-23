import torch

from utils import save_model, load_model
from models.ops import OP_REGISTRY
from models.resnet import ResNet, BasicBlock, CIFARStem, ImageNetStem
from models.workload_factory import resnet_configs, generate_random_config


def test_resnet_cifar10():
    skeleton = resnet_configs["resnet18"]
    config = generate_random_config(
        skeleton.channels, skeleton.blocks, skeleton.strides, OP_REGISTRY
    )

    net = ResNet(BasicBlock, skeleton.blocks, config, num_classes=10, stem=CIFARStem)
    data = torch.rand((1, 3, 32, 32))

    y = net(data)

    assert y.size() == torch.Size((1, 10))


def test_resnet_cifar100():
    skeleton = resnet_configs["resnet18"]
    config = generate_random_config(
        skeleton.channels, skeleton.blocks, skeleton.strides, OP_REGISTRY
    )

    net = ResNet(BasicBlock, skeleton.blocks, config, num_classes=100, stem=CIFARStem)
    data = torch.rand((1, 3, 32, 32))

    y = net(data)

    assert y.size() == torch.Size((1, 100))


def test_resnet_imagenet():
    skeleton = resnet_configs["resnet18"]
    config = generate_random_config(
        skeleton.channels, skeleton.blocks, skeleton.strides, OP_REGISTRY
    )

    net = ResNet(
        BasicBlock, skeleton.blocks, config, num_classes=1000, stem=ImageNetStem
    )
    data = torch.rand((1, 3, 224, 224))

    y = net(data)

    assert y.size() == torch.Size((1, 1000))


def test_save_and_load():
    skeleton = resnet_configs["resnet18"]
    config = generate_random_config(
        skeleton.channels, skeleton.blocks, skeleton.strides, OP_REGISTRY
    )

    net = ResNet(BasicBlock, skeleton.blocks, config, num_classes=10, stem=CIFARStem)
    data = torch.rand((1, 3, 32, 32))

    y1 = net(data)

    import tempfile

    with tempfile.NamedTemporaryFile() as tmp:
        print(tmp.name)

        save_model(net, tmp.name, save_weights=True)

        net2 = load_model(tmp.name)

    y2 = net2(data)

    assert y1.size() == y2.size()
