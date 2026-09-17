"""Tests for the deep architecture factory.

The central test is ``test_only_the_head_changed``. Replacing a
classification head is the kind of operation that fails quietly: swap one
layer too many and the model still builds, still trains, and still reports
accuracy - just with part of a pretrained backbone thrown away. Nothing
raises, and the only symptom is a number that is lower than it should be by an
amount nobody can attribute. So rather than trusting that six different head
locations were each written down correctly, the test rebuilds each
architecture with its original 1000-way head and asserts the parameter count
outside the head is unchanged.

Everything here builds with ``pretrained=False`` on purpose. The real runs use
ImageNet weights, but downloading 1.1 GB of checkpoints is not something a
test suite should do on a fresh clone, and none of these assertions are about
the weight *values* - they are about structure, which is identical either way.
"""

import pytest

from samuel_collins_cv_benchmarking import deep_models
from samuel_collins_cv_benchmarking._config import (
    CNN_ARCHITECTURES,
    CNN_DEVICE_PREFERENCE,
)

# Deliberately not 10, and not any internal layer width in these nine
# networks, so "the Linear with this many outputs" identifies the head alone.
CLASSES = 7

# Constructor arguments the reference model needs in order to be the *same*
# architecture the factory produces. Only GoogLeNet needs one: the factory
# disables its auxiliary classifiers so that a single training loop can serve
# all nine architectures, so a reference built with them still attached would
# differ by the aux branches - a real difference, deliberately introduced, and
# not the head-substitution error this test is looking for.
REFERENCE_KWARGS = {"googlenet": {"aux_logits": False}}


def linear_params_with_width(model, width: int) -> int:
    """Parameters of every Linear whose output is exactly ``width`` wide."""
    import torch.nn as nn

    return sum(
        parameter.numel()
        for module in model.modules()
        if isinstance(module, nn.Linear) and module.out_features == width
        for parameter in module.parameters()
    )


@pytest.mark.parametrize("key", deep_models.architecture_names())
def test_head_has_the_requested_width(key):
    """Section 8: the final classification layer predicts the dataset's classes."""
    import torch.nn as nn

    model = deep_models.get_model(key, num_classes=CLASSES, pretrained=False)
    heads = [module for module in model.modules()
             if isinstance(module, nn.Linear) and module.out_features == CLASSES]
    assert len(heads) == 1, f"{key}: expected exactly one {CLASSES}-way head"


@pytest.mark.parametrize("key", deep_models.architecture_names())
def test_only_the_head_changed(key):
    """The backbone must be untouched by the head substitution.

    Compares against the architecture as torchvision builds it, with its
    original 1000-way ImageNet head. If the two agree on every parameter
    outside the head, the registry entry for this architecture named the right
    layer; if the registry pointed one layer too deep, this is where it shows.
    """
    import torchvision.models as tvm

    architecture = deep_models.BY_KEY[key]
    reference = getattr(tvm, architecture.constructor)(
        weights=None, **REFERENCE_KWARGS.get(key, {}))
    ours = deep_models.get_model(key, num_classes=CLASSES, pretrained=False)

    reference_backbone = (
        sum(p.numel() for p in reference.parameters())
        - linear_params_with_width(reference, 1000)
    )
    our_backbone = (
        sum(p.numel() for p in ours.parameters())
        - linear_params_with_width(ours, CLASSES)
    )
    assert our_backbone == reference_backbone


@pytest.mark.parametrize("key", deep_models.architecture_names())
def test_forward_returns_one_logit_tensor_in_training_mode(key):
    """One training loop serves all nine, so all nine must return a tensor.

    GoogLeNet is the reason this test is parametrized rather than written
    once: pretrained GoogLeNet returns a namedtuple of three tensors in
    training mode because of its auxiliary classifiers, and the factory
    disables them precisely so this assertion holds everywhere.
    """
    import torch

    model = deep_models.get_model(key, num_classes=CLASSES, pretrained=False)
    model.train()
    output = model(torch.randn(2, 3, 224, 224))

    assert isinstance(output, torch.Tensor), f"{key} returned {type(output).__name__}"
    assert output.shape == (2, CLASSES)


def test_googlenet_auxiliary_classifiers_are_gone():
    """Stated directly, because it is a documented deviation from the paper."""
    model = deep_models.get_model("googlenet", num_classes=CLASSES, pretrained=False)
    assert model.aux_logits is False
    assert model.aux1 is None
    assert model.aux2 is None


def test_googlenet_aux_removal_costs_the_parameters_the_report_claims():
    """Pin the size of the deviation, since the report quotes it.

    Removing the two auxiliary branches drops 4,329,984 parameters from
    GoogLeNet - the branches minus their own 1000-way heads. The report
    describes this as a deviation from the paper's training recipe, and a
    number in a report should be a number something checks.
    """
    import torch.nn as nn
    import torchvision.models as tvm

    with_aux = tvm.googlenet(weights=None)
    aux_total = sum(parameter.numel() for name, parameter
                    in with_aux.named_parameters() if name.startswith("aux"))
    aux_heads = sum(
        parameter.numel()
        for branch in (with_aux.aux1, with_aux.aux2)
        for module in branch.modules()
        if isinstance(module, nn.Linear) and module.out_features == 1000
        for parameter in module.parameters()
    )
    assert aux_total - aux_heads == 4_329_984


# -- parameter counting ------------------------------------------------------

@pytest.mark.parametrize("key", deep_models.architecture_names())
def test_full_fine_tune_leaves_every_parameter_trainable(key):
    """Section 17 wants both counts; under section 8's fine-tune they agree."""
    model = deep_models.get_model(key, num_classes=CLASSES, pretrained=False)
    counts = deep_models.parameter_counts(model)

    assert counts["total_parameters"] == counts["trainable_parameters"]
    assert counts["frozen_parameters"] == 0
    assert counts["total_parameters"] > 0


def test_parameter_counts_reports_millions_consistently():
    model = deep_models.get_model("resnet18", num_classes=CLASSES, pretrained=False)
    counts = deep_models.parameter_counts(model)
    assert counts["total_parameters_millions"] == round(
        counts["total_parameters"] / 1e6, 3)


# -- metadata, for sections 23, 25 and 26 -----------------------------------

def test_every_architecture_carries_its_report_metadata():
    for key in deep_models.architecture_names():
        metadata = deep_models.architecture_metadata(key)
        assert metadata["family"], f"{key} has no family for the section 23 column"
        assert metadata["year"] > 2000
        assert metadata["ideas"], f"{key} has no design ideas for section 25"
        assert metadata["addressed"], f"{key} has no timeline note for section 26"


def test_architectures_are_ordered_oldest_first():
    """Section 26's timeline should fall out of the registry order, not a re-sort."""
    years = [deep_models.BY_KEY[key].year
             for key in deep_models.architecture_names()]
    assert years == sorted(years)


def test_registry_matches_the_configured_architecture_list():
    """_config lists ten; nine are built here and YOLO is the tenth."""
    configured = set(CNN_ARCHITECTURES)
    built = set(deep_models.architecture_names())
    assert built < configured
    assert configured - built == {"yolo_cls"}


def test_weights_identifier_names_the_resolved_checkpoint():
    """DEFAULT moves over time, so the report records what it resolved to."""
    identifier = deep_models.weights_identifier("resnet50")
    assert "IMAGENET1K" in identifier
    assert deep_models.weights_identifier("yolo_cls") == "n/a"


# -- the refusals and the device --------------------------------------------

def test_unknown_architecture_lists_the_valid_names():
    with pytest.raises(KeyError) as error:
        deep_models.get_model("resnet101", num_classes=CLASSES)
    assert "resnet50" in str(error.value)


def test_yolo_is_not_constructible_here():
    """It runs as a subprocess in its own environment; the error should say so."""
    with pytest.raises(KeyError, match="subprocess"):
        deep_models.get_model("yolo_cls", num_classes=CLASSES)


def test_device_is_one_of_the_preferred_devices():
    import torch

    device = deep_models.device_for()
    assert isinstance(device, torch.device)
    assert device.type in CNN_DEVICE_PREFERENCE


def test_cpu_is_always_reachable_as_a_fallback():
    """A machine with no accelerator must still run the benchmark."""
    assert deep_models.device_for(("cpu",)).type == "cpu"
