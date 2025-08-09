import torch._prims_common as prims_common  # type: ignore

from tensorcontainer.types import Shape


def test_shape_alias_matches_torch_prims_common():
    assert Shape == prims_common.ShapeType, (
        "tensorcontainer.types.Shape diverges from torch._prims_common.ShapeType"
    )
