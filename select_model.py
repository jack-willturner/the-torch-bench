import torch
import argparse
import numpy as np

from utils import save_model, load_model
from models.ops import OP_REGISTRY
from models.resnet import ResNet, BasicBlock, CIFARStem, ImageNetStem
from models.workload_factory import resnet_configs, generate_random_config
from proxy import NASWOT
from gymnastics.datasets import get_data_loaders

from tqdm import tqdm

parser = argparse.ArgumentParser(description="Torch bench")
parser.add_argument(
    "--dataset",
    default="CIFAR10",
    type=str,
    help="Path to actual dataset",
)
parser.add_argument(
    "--path_to_data",
    default="~/datasets",
    type=str,
    help="Path to actual dataset",
)
parser.add_argument(
    "--num_trials",
    default=100,
    type=int,
    help="How many configs to try",
)

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if args.dataset == "CIFAR10":
    stem = CIFARStem
    num_classes = 10
elif args.dataset == "ImageNet":
    stem = ImageNetStem
    num_classes = 1000

skeleton = resnet_configs["resnet18"]

proxy = NASWOT()
data_loader = get_data_loaders(
    args.dataset,
    args.path_to_data,
    batch_size=128,
)

configs = []
scores = []

for trial in tqdm(range(args.num_trials)):

    config = generate_random_config(
        skeleton.channels, skeleton.blocks, skeleton.strides, OP_REGISTRY
    )

    net = ResNet(BasicBlock, skeleton.blocks, config, num_classes=1, stem=ImageNetStem)
    minibatch, target = data_loader.sample_minibatch()

    minibatch = minibatch.to(device)
    net = net.to(device)
    target = target.to(device)
    score: float = proxy.score(net, minibatch, target)

    configs.append(config)
    scores.append(score)

best_config = configs[np.argmax(scores)]
net = ResNet(BasicBlock, skeleton.blocks, config, num_classes=num_classes, stem=stem)

save_model(net, "configs/best_model.t7")
