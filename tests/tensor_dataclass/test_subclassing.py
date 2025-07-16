import torch
import inspect

from tensorcontainer.tensor_dataclass import TensorDataClass


class TestInitSignature:
    def test_single_parent_class(self):
        class A(TensorDataClass):
            x: torch.Tensor

        class B(A):
            y: str

        # 1. Test __init__ signature
        actual_signature = inspect.signature(B.__init__)

        # Create expected signature with proper parameters and type annotations
        from typing import Optional

        expected_signature = inspect.Signature(
            [
                inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                inspect.Parameter(
                    "shape",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=torch.Size,
                ),
                inspect.Parameter(
                    "device",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=Optional[torch.device],
                ),
                inspect.Parameter(
                    "x",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=torch.Tensor,
                ),
                inspect.Parameter(
                    "y", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str
                ),
            ],
            return_annotation=None,
        )

        assert actual_signature == expected_signature

    def test_multiple_parent_classes_multiple_tensor_data_class(self):
        class A(TensorDataClass):
            x: torch.Tensor

        class B(TensorDataClass):
            y: str

        class C(B, A):
            z: int

        # 1. Test __init__ signature
        actual_signature = inspect.signature(C.__init__)

        # Create expected signature with proper parameters and type annotations
        from typing import Optional

        expected_signature = inspect.Signature(
            [
                inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                inspect.Parameter(
                    "shape",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=torch.Size,
                ),
                inspect.Parameter(
                    "device",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=Optional[torch.device],
                ),
                inspect.Parameter(
                    "x",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=torch.Tensor,
                ),
                inspect.Parameter(
                    "y", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str
                ),
                inspect.Parameter(
                    "z", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=int
                ),
            ],
            return_annotation=None,
        )

        assert actual_signature == expected_signature

    def test_multiple_parent_classes_single_tensor_data_class(self):
        class A(TensorDataClass):
            x: torch.Tensor

        class B:
            y: str

        class C(A, B):
            z: int

        # 1. Test __init__ signature
        actual_signature = inspect.signature(C.__init__)

        # Create expected signature with proper parameters and type annotations
        from typing import Optional

        expected_signature = inspect.Signature(
            [
                inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                inspect.Parameter(
                    "shape",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=torch.Size,
                ),
                inspect.Parameter(
                    "device",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=Optional[torch.device],
                ),
                inspect.Parameter(
                    "x",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=torch.Tensor,
                ),
                inspect.Parameter(
                    "z", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=int
                ),
            ],
            return_annotation=None,
        )

        assert actual_signature == expected_signature
