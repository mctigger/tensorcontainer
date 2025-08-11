import torch._prims_common as prims_common  # type: ignore

from tensorcontainer.types import DeviceLike, ShapeLike


def test_shape_alias_matches_torch_prims_common():
    assert ShapeLike == prims_common.ShapeType, (
        "tensorcontainer.types.Shape diverges from torch._prims_common.ShapeType"
    )


def test_device_alias_matches_torch_prims_common():
    assert DeviceLike == prims_common.DeviceLikeType
