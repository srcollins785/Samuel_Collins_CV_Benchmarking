"""The model factory: nine pretrained architectures, one interface.

Section 10 asks for a single ``get_model(name, num_classes, pretrained)`` and
section 8 asks the framework to replace the final classification layer
automatically. The awkward fact underneath both requirements is that these
nine architectures keep their classifier in six different places - ``fc`` on
the ResNets and GoogLeNet, a bare ``classifier`` on DenseNet, and index 6, 1,
2 or 3 of a ``classifier`` Sequential on the rest.

The temptation is to write something clever: walk the module graph, find the
last ``nn.Linear``, swap it. That works today and fails silently the day
torchvision reorders a Sequential, because a wrong-but-plausible substitution
produces a model that trains and reports numbers rather than one that raises.
So every head location is written down explicitly in ``ARCHITECTURES`` below.
``get_model`` is still automatic from the caller's side; the difference is
that the substitution is auditable, and a test checks for all nine that the
new head has the right output width *and* that the backbone parameter count
did not move - which is what catches a swap that replaced more than the head.

The registry also carries the family, year and design ideas each architecture
contributed. Sections 23, 25 and 26 all need that metadata - the family column
in the master table, the architecture explanations, the historical timeline -
and keeping it here means the report generator reads it from the artifacts
this package writes rather than holding a second copy that can disagree.

YOLO is deliberately absent. It cannot be constructed in this environment at
all (see ``_config.YOLO_MODEL``), so listing it here would imply otherwise.
"""

from dataclasses import dataclass, field
from typing import Callable

from ._config import (
    CNN_DEVICE_PREFERENCE,
    CNN_FREEZE_BACKBONE,
    CNN_PRETRAINED,
    RANDOM_SEED,
)


# -- head replacement --------------------------------------------------------
#
# Two shapes cover all nine: the head is an attribute, or it is one entry of a
# Sequential held in an attribute. Both read ``in_features`` off the layer they
# are replacing rather than hard-coding a width, so the same entry keeps
# working if torchvision changes a backbone's output dimension.

def _head_attribute(attribute: str) -> Callable:
    """Replace ``model.<attribute>``, itself a Linear."""
    def replace(model, num_classes: int):
        import torch.nn as nn

        existing = getattr(model, attribute)
        setattr(model, attribute, nn.Linear(existing.in_features, num_classes))
    return replace


def _head_index(attribute: str, position: int) -> Callable:
    """Replace ``model.<attribute>[position]``, a Linear inside a Sequential."""
    def replace(model, num_classes: int):
        import torch.nn as nn

        block = getattr(model, attribute)
        block[position] = nn.Linear(block[position].in_features, num_classes)
    return replace


def _googlenet_head(model, num_classes: int) -> None:
    """Replace GoogLeNet's head and disable its auxiliary classifiers.

    Pretrained GoogLeNet ships the two auxiliary heads from the original
    paper, which make ``forward()`` return a namedtuple of three logit tensors
    in training mode instead of one tensor. Every other architecture returns a
    tensor, so leaving them on would mean a special case inside the shared
    training loop - and a shared training loop is the thing that makes these
    numbers comparable at all.

    Disabling them drops GoogLeNet's auxiliary loss, which is a real deviation
    from the paper's training recipe and is recorded as such in the report. It
    costs a little of the regularization the aux heads provided; it does not
    change the network's inference-time architecture, since the aux branches
    are discarded at inference anyway.
    """
    model.aux_logits = False
    model.aux1 = None
    model.aux2 = None
    _head_attribute("fc")(model, num_classes)


@dataclass(frozen=True)
class Architecture:
    """One architecture's constructor, head location and provenance."""

    key: str
    name: str
    # Section 23's Family column.
    family: str
    # Publication year, so section 26's timeline orders itself from data.
    year: int
    # torchvision constructor name, resolved lazily so importing this module
    # does not require torchvision.
    constructor: str
    replace_head: Callable
    # Section 25's "main ideas behind each architecture", as short phrases the
    # report expands rather than prose duplicated in the report generator.
    ideas: tuple = field(default_factory=tuple)
    # The design limitation this generation set out to fix, for section 26.
    addressed: str = ""


ARCHITECTURES = (
    Architecture(
        key="alexnet", name="AlexNet", family="Deep CNN", year=2012,
        constructor="alexnet", replace_head=_head_index("classifier", 6),
        ideas=("ReLU activations instead of saturating nonlinearities",
               "dropout in the fully connected layers",
               "large early convolution kernels (11x11 stride 4)",
               "GPU training at ImageNet scale"),
        addressed="Showed that deep CNNs beat hand-engineered features, which "
                  "is the result that ended the feature-engineering era.",
    ),
    Architecture(
        key="vgg16", name="VGG16", family="Deep CNN", year=2014,
        constructor="vgg16", replace_head=_head_index("classifier", 6),
        ideas=("stacked 3x3 convolutions in place of large kernels",
               "uniform, very deep sequential design",
               "very large parameter count concentrated in the dense layers"),
        addressed="Replaced AlexNet's large kernels with stacks of small ones, "
                  "gaining depth and receptive field at lower parameter cost "
                  "per layer - but ballooning the fully connected head.",
    ),
    Architecture(
        key="googlenet", name="GoogLeNet", family="Multi-branch CNN", year=2014,
        constructor="googlenet", replace_head=_googlenet_head,
        ideas=("Inception modules with parallel branches",
               "several receptive-field sizes in one layer",
               "1x1 convolutions as cheap dimensionality reduction",
               "global average pooling instead of a huge dense head"),
        addressed="Attacked VGG's parameter cost directly: width and multi-scale "
                  "features instead of depth alone, at a fraction of the parameters.",
    ),
    Architecture(
        key="resnet18", name="ResNet18", family="Residual CNN", year=2015,
        constructor="resnet18", replace_head=_head_attribute("fc"),
        ideas=("residual connections",
               "identity skip paths that preserve gradient flow",
               "batch normalization throughout"),
        addressed="Solved the degradation problem: past roughly twenty layers, "
                  "plain deep networks got *worse*, and skip connections made "
                  "depth trainable again.",
    ),
    Architecture(
        key="resnet50", name="ResNet50", family="Residual CNN", year=2015,
        constructor="resnet50", replace_head=_head_attribute("fc"),
        ideas=("bottleneck residual blocks (1x1, 3x3, 1x1)",
               "greater depth at controlled parameter cost",
               "the same identity skip paths as ResNet18"),
        addressed="Shows what depth buys inside one architectural family, which "
                  "is why the assignment pairs it with ResNet18.",
    ),
    Architecture(
        key="densenet121", name="DenseNet121", family="Dense CNN", year=2016,
        constructor="densenet121", replace_head=_head_attribute("classifier"),
        ideas=("dense connectivity - every layer sees all earlier feature maps",
               "feature reuse rather than feature relearning",
               "narrow layers, so parameters stay low despite the connectivity"),
        addressed="Took ResNet's skip idea further: concatenate rather than add, "
                  "so features are reused instead of recomputed.",
    ),
    Architecture(
        key="mobilenet_v3_large", name="MobileNetV3-Large",
        family="Efficient CNN", year=2019,
        constructor="mobilenet_v3_large", replace_head=_head_index("classifier", 3),
        ideas=("depthwise separable convolutions",
               "inverted residuals with linear bottlenecks",
               "squeeze-and-excitation attention",
               "architecture search tuned for mobile latency, not just FLOPs"),
        addressed="Reframed the goal from accuracy to accuracy per millisecond on "
                  "a phone, which is the question this benchmark's deployment "
                  "discussion actually asks.",
    ),
    Architecture(
        key="efficientnet_b0", name="EfficientNet-B0",
        family="Scaled CNN", year=2019,
        constructor="efficientnet_b0", replace_head=_head_index("classifier", 1),
        ideas=("compound scaling of depth, width and resolution together",
               "mobile inverted bottleneck blocks",
               "a scaling rule rather than a hand-tuned family"),
        addressed="Showed that depth, width and input resolution should be scaled "
                  "jointly - scaling one alone saturates.",
    ),
    Architecture(
        key="convnext_tiny", name="ConvNeXt-Tiny",
        family="Modern CNN", year=2022,
        constructor="convnext_tiny", replace_head=_head_index("classifier", 2),
        ideas=("transformer-inspired design kept fully convolutional",
               "large 7x7 depthwise kernels",
               "LayerNorm in place of BatchNorm, GELU in place of ReLU",
               "inverted bottleneck with fewer, wider blocks"),
        addressed="Answered whether vision transformers won because of attention "
                  "or because of their training recipe and design choices - a "
                  "modernized CNN matches them.",
    ),
)

BY_KEY = {architecture.key: architecture for architecture in ARCHITECTURES}


def architecture_names() -> tuple:
    """The nine constructible architectures, oldest first."""
    return tuple(architecture.key for architecture in ARCHITECTURES)


def architecture_metadata(key: str) -> dict:
    """Family, year, ideas and timeline note for one architecture.

    Written into the run artifacts so the report can produce sections 23, 25
    and 26 from data rather than from a second copy of this information.
    """
    architecture = BY_KEY[key]
    return {
        "key": architecture.key,
        "name": architecture.name,
        "family": architecture.family,
        "year": architecture.year,
        "ideas": list(architecture.ideas),
        "addressed": architecture.addressed,
    }


def device_for(preference: tuple = CNN_DEVICE_PREFERENCE):
    """The best available device, in preference order.

    CUDA is checked first even though this machine has none, so the same code
    reproduces on a CUDA host without an edit. Whichever device is chosen is
    recorded in the run configuration, because an inference benchmark without
    the hardware it ran on is not a measurement (section 20).
    """
    import torch

    for name in preference:
        if name == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if name == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if name == "cpu":
            return torch.device("cpu")
    return torch.device("cpu")


def weights_identifier(model_name: str) -> str:
    """Which pretrained weights ``DEFAULT`` actually resolves to.

    ``weights="DEFAULT"`` is a moving target - torchvision repoints it when
    better weights land, and ResNet50's DEFAULT is already IMAGENET1K_V2 rather
    than the V1 weights of the original paper. Recording the resolved enum
    means the report says which weights produced its numbers instead of
    implying the question has one permanent answer.
    """
    import torchvision.models as tvm

    if model_name not in BY_KEY:
        return "n/a"
    try:
        return str(tvm.get_model_weights(BY_KEY[model_name].constructor).DEFAULT)
    except Exception:
        return "unknown"


def get_model(model_name: str, num_classes: int = 10, pretrained: bool = CNN_PRETRAINED):
    """Build one architecture with a fresh ``num_classes``-way head.

    Parameters
    ----------
    model_name
        One of :func:`architecture_names`.
    num_classes
        Output width. The assignment's example says 10 because CIFAR-10 has
        ten classes; here it is however many classes the dataset actually has,
        which the caller passes in rather than assuming.
    pretrained
        Load ImageNet weights (section 8). ``False`` trains from scratch,
        which is not the primary experiment but makes a from-scratch
        comparison possible without a second code path.

    Returns
    -------
    torch.nn.Module
        Ready to train, every parameter trainable unless
        ``CNN_FREEZE_BACKBONE`` is set.

    Raises
    ------
    KeyError
        Unknown architecture, listing the valid names.
    """
    import torch
    import torchvision.models as tvm

    if model_name not in BY_KEY:
        raise KeyError(
            f"Unknown architecture {model_name!r}. Available: "
            f"{', '.join(architecture_names())}. YOLO classification is not "
            "built here - it runs as a separate subprocess."
        )

    architecture = BY_KEY[model_name]

    # The replacement head is randomly initialized, so seed before building.
    # Without this the head's starting weights differ between runs and two
    # runs of the same architecture are not comparable to each other.
    torch.manual_seed(RANDOM_SEED)

    constructor = getattr(tvm, architecture.constructor)
    model = constructor(weights="DEFAULT" if pretrained else None)
    architecture.replace_head(model, num_classes)

    if CNN_FREEZE_BACKBONE:
        # Not the configured path, but if it is ever switched on, freeze
        # everything and then thaw the head - rather than trying to name the
        # head's parameters, which differ across these nine architectures.
        for parameter in model.parameters():
            parameter.requires_grad = False
        architecture.replace_head(model, num_classes)

    return model


def parameter_counts(model) -> dict:
    """Total and trainable parameter counts (section 17).

    Under the configured full fine-tune these are equal. That is worth showing
    in the table rather than asserting away: a reader comparing against a
    frozen-backbone benchmark elsewhere needs to see which regime produced
    these numbers, and two identical columns say it plainly.
    """
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad)
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "total_parameters_millions": round(total / 1e6, 3),
        "frozen_parameters": int(total - trainable),
    }
