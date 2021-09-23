from .resnet import ResNet

def get_model(model):
    if model == "resnet":
        return ResNet
    else:
        raise ValueError("Invalid model type")
