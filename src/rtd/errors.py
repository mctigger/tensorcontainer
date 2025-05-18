from torch import Tensor


class ShapeMismatchError(ValueError):
    """Raised when tensor shapes are incompatible for the requested operation."""

    def __init__(self, message: str, tensor: Tensor):
        # initialize the base ValueError with your message
        super().__init__(message)
        # store the tensor that caused the error
        self.tensor = tensor
