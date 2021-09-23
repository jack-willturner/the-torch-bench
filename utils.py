import torch
from models import get_model


def save_model(model, filename, save_weights=False):
    torch.save(
        {
            "model_class": model.model_class,
            "state_dict": model.state_dict() if save_weights else None,
            "block": model.block,
            "num_blocks": model.num_blocks,
            "configs": model.configs,
            "num_classes": model.num_classes,
            "stem": model.stem_type,
        },
        filename,
    )


def load_model(checkpoint):
    ckpt = torch.load(checkpoint)

    ## get the model base class
    model_class = get_model(ckpt["model_class"])

    ## instantiate a model
    model = model_class(
        ckpt["block"],
        ckpt["num_blocks"],
        ckpt["configs"],
        ckpt["num_classes"],
        ckpt["stem"],
    )

    ## load weights if there are any
    if ckpt["state_dict"]:
        model.load_state_dict(ckpt["state_dict"])

    return model
