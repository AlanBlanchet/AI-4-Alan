from pathlib import Path
from typing import Any, Optional, Union

import torch
from einops import rearrange
from PIL.Image import SupportsArrayInterface
from typing_extensions import Self

from myconf import Cast, Consumed
from myconf.config import resolve_metaclass

from ..configs.base import Base


class TensorPath(Path):
    """A Path subclass with tensor loading capabilities for files and URLs"""

    SUPPORTED_EXTENSIONS = {".pt", ".pth", ".npy", ".npz"}

    def __new__(cls, path_arg):
        # Handle URLs differently from file paths
        if isinstance(path_arg, str) and (
            path_arg.startswith("http://") or path_arg.startswith("https://")
        ):
            # For URLs, create a simple object that behaves like a path
            instance = object.__new__(cls)
            instance._url = path_arg
            instance._is_url = True
            return instance

        # Handle regular file paths
        path_instance = Path(path_arg)

        # Validate file paths
        if (
            not path_instance.exists()
            or path_instance.suffix not in cls.SUPPORTED_EXTENSIONS
        ):
            raise ValueError(f"Invalid tensor path: {path_instance}")

        # Create our instance by copying the path
        instance = object.__new__(cls)
        instance._is_url = False

        # Copy all the internal attributes from the path instance
        for attr in dir(path_instance):
            if attr.startswith("_") and hasattr(path_instance, attr):
                try:
                    setattr(instance, attr, getattr(path_instance, attr))
                except (AttributeError, TypeError):
                    pass

        return instance

    def __str__(self):
        if hasattr(self, "_is_url") and self._is_url:
            return self._url
        return (
            super().__str__()
            if hasattr(super(), "__str__")
            else str(self._url if hasattr(self, "_url") else "unknown")
        )

    def __repr__(self):
        if hasattr(self, "_is_url") and self._is_url:
            return f"TensorPath({self._url!r})"
        return f"TensorPath({str(self)!r})"

    @property
    def suffix(self):
        if hasattr(self, "_is_url") and self._is_url:
            return ""  # URLs don't have file extensions in the traditional sense
        return super().suffix

    def exists(self):
        if hasattr(self, "_is_url") and self._is_url:
            return True  # Assume URLs exist (will be validated on load)
        return super().exists()

    def load_tensor(self):
        """Load the tensor from the file path or URL"""
        if hasattr(self, "_is_url") and self._is_url:
            # Handle URL image loading
            try:
                import io

                import requests
                from PIL import Image

                # Download the image
                response = requests.get(self._url, timeout=30)
                response.raise_for_status()

                # Open image with PIL
                image = Image.open(io.BytesIO(response.content))

                # Convert to RGB if necessary
                if image.mode != "RGB":
                    image = image.convert("RGB")

                # Convert to tensor (H, W, C) -> (C, H, W)
                import numpy as np

                image_array = np.array(image)
                tensor = (
                    torch.from_numpy(image_array).float() / 255.0
                )  # Normalize to [0, 1]
                tensor = tensor.permute(2, 0, 1)  # Change from HWC to CHW format

                return tensor

            except ImportError as e:
                missing_lib = str(e).split()[-1].strip("'")
                raise ImportError(
                    f"Missing required library for URL image loading: {missing_lib}. Install with: pip install {missing_lib}"
                )
            except Exception as e:
                raise ValueError(f"Failed to load image from URL {self._url}: {e}")
        else:
            # Handle file loading
            if self.suffix in [".pt", ".pth"]:
                return torch.load(self)
            elif self.suffix in [".npy", ".npz"]:
                import numpy as np

                if self.suffix == ".npy":
                    return torch.from_numpy(np.load(self))
                else:  # .npz
                    npz_data = np.load(self)
                    arrays = list(npz_data.values())
                    if arrays:
                        return torch.from_numpy(arrays[0])
                    else:
                        raise ValueError(f"Empty .npz file: {self}")
            else:
                raise ValueError(f"Unsupported tensor file format: {self.suffix}")

    def __torch_tensor__(self):
        """Enable automatic conversion to torch.Tensor"""
        return self.load_tensor()

    def __call__(self, *args, **kwargs):
        """Make TensorPath callable to return the loaded tensor"""
        return self.load_tensor()


class TensorBase(
    Base,
    torch.Tensor,
    SupportsArrayInterface,
    metaclass=resolve_metaclass(Base, torch.Tensor, SupportsArrayInterface),
):
    data: Consumed[Cast[Union[Cast[str, TensorPath], list, Any], torch.Tensor]] = None
    device: Consumed[str] = None
    dtype: Consumed[torch.dtype] = None
    requires_grad: Consumed[bool] = False

    @staticmethod
    def __new__(cls, data: Any, **kwargs):
        device = kwargs.get("device")
        requires_grad = kwargs.get("requires_grad", False)

        # At this point, data should already be converted to torch.Tensor by MyConf Cast
        if not isinstance(data, torch.Tensor):
            # Fallback for any unconverted data
            data = torch.tensor(data)

        # Smart dtype handling: preserve input tensor's dtype if no explicit dtype provided
        if "dtype" in kwargs:
            # Explicit dtype provided - use it
            dtype = kwargs["dtype"]
            tensor = data.to(device=device, dtype=dtype)
        else:
            # No explicit dtype - preserve input tensor's dtype, only change device if needed
            tensor = data.to(device=device)

        if requires_grad:
            tensor = tensor.requires_grad_(True)

        tensor = cls.tensor_format(tensor, kwargs)

        return torch.Tensor._make_subclass(cls, tensor)

    @classmethod
    def tensor_format(
        cls, tensor: torch.Tensor, kwargs: dict[str, Any]
    ) -> torch.Tensor:
        return tensor

    @classmethod
    def _make_subclass_efficient(cls, tensor: torch.Tensor) -> Self:
        """Create subclass efficiently without going through __new__ and tensor_format"""
        return torch.Tensor._make_subclass(cls, tensor)

    def _base_from_tensor(self, tensor: torch.Tensor) -> "TensorBase":
        """Create a TensorBase from tensor, losing specialized type"""
        return torch.Tensor._make_subclass(TensorBase, tensor)

    def safe(self):
        return self.detach().cpu()

    def is_int(self) -> bool:
        return self.dtype == torch.int

    def is_float(self) -> bool:
        return self.dtype == torch.float

    def is_bool(self) -> bool:
        return self.dtype == torch.bool

    def rearrange(self, pattern: str) -> Self:
        result = rearrange(self, pattern)
        return self.__class__(result)

    @property
    def __array_interface__(self):
        numpy_arr = self.numpy()
        interface = numpy_arr.__array_interface__.copy()
        if interface.get("strides") is None:
            interface["strides"] = numpy_arr.strides
        return interface

    def __array__(self):
        return self.numpy()

    def tobytes(self):
        return self.numpy().tobytes()

    def numpy(self):
        safe_tensor = self.safe()
        if hasattr(safe_tensor, "data") and hasattr(safe_tensor, "dtype"):
            plain_tensor = torch.tensor(
                safe_tensor.data, dtype=safe_tensor.dtype, device=safe_tensor.device
            )
            return plain_tensor.numpy()
        else:
            return safe_tensor.detach().cpu().numpy()

    def __mul__(self, other) -> "Self":
        result = super().__mul__(other)
        return self.__class__(result)

    def __rmul__(self, other) -> "Self":
        result = super().__rmul__(other)
        return self.__class__(result)

    def __add__(self, other) -> "Self":
        result = super().__add__(other)
        return self.__class__(result)

    def __radd__(self, other) -> "Self":
        result = super().__radd__(other)
        return self.__class__(result)

    def __sub__(self, other) -> "Self":
        result = super().__sub__(other)
        return self.__class__(result)

    def __rsub__(self, other) -> "Self":
        result = super().__rsub__(other)
        return self.__class__(result)

    def __truediv__(self, other) -> "Self":
        result = super().__truediv__(other)
        return self.__class__(result)

    def __rtruediv__(self, other) -> "Self":
        result = super().__rtruediv__(other)
        return self.__class__(result)

    def __matmul__(self, other) -> "Self":
        result = super().__matmul__(other)
        return self.__class__(result)

    def __rmatmul__(self, other) -> "Self":
        result = super().__rmatmul__(other)
        return self.__class__(result)

    def mul(self, other) -> "Self":
        result = super().mul(other)
        return self.__class__(result)

    def add(self, other) -> "Self":
        result = super().add(other)
        return self.__class__(result)

    def sub(self, other) -> "Self":
        result = super().sub(other)
        return self.__class__(result)

    def div(self, other) -> "Self":
        result = super().div(other)
        return self.__class__(result)

    def matmul(self, other) -> "Self":
        result = super().matmul(other)
        return self.__class__(result)

    def __eq__(self, other):
        if isinstance(other, torch.Tensor):
            return torch.equal(self, other)
        return False

    def __str__(self):
        as_tensor = self.as_tensor()
        return str(as_tensor)

    def __repr__(self):
        as_tensor = self.as_tensor()
        return f"TensorBase({as_tensor})"

    def as_tensor(self):
        return torch.tensor(self.tolist(), dtype=self.dtype, device=self.device)

    def __floordiv__(self, other: Union[int, float, torch.Tensor]) -> Self:
        return self.__class__(super().__floordiv__(other))

    def __rfloordiv__(self, other: Union[int, float, torch.Tensor]) -> Self:
        return self.__class__(super().__rfloordiv__(other))

    def __mod__(self, other: Union[int, float, torch.Tensor]) -> Self:
        return self.__class__(super().__mod__(other))

    def __rmod__(self, other: Union[int, float, torch.Tensor]) -> Self:
        return self.__class__(super().__rmod__(other))

    def __pow__(self, other: Union[int, float, torch.Tensor]) -> Self:
        return self.__class__(super().__pow__(other))

    def __rpow__(self, other: Union[int, float, torch.Tensor]) -> Self:
        return self.__class__(super().__rpow__(other))

    def __and__(self, other: torch.Tensor) -> Self:
        return self.__class__(super().__and__(other))

    def __rand__(self, other: torch.Tensor) -> Self:
        return self.__class__(super().__rand__(other))

    def __or__(self, other: torch.Tensor) -> Self:
        return self.__class__(super().__or__(other))

    def __ror__(self, other: torch.Tensor) -> Self:
        return self.__class__(super().__ror__(other))

    def __xor__(self, other: torch.Tensor) -> Self:
        return self.__class__(super().__xor__(other))

    def __rxor__(self, other: torch.Tensor) -> Self:
        return self.__class__(super().__rxor__(other))

    def __lshift__(self, other: Union[int, torch.Tensor]) -> Self:
        return self.__class__(super().__lshift__(other))

    def __rlshift__(self, other: Union[int, torch.Tensor]) -> Self:
        return self.__class__(super().__rlshift__(other))

    def __rshift__(self, other: Union[int, torch.Tensor]) -> Self:
        return self.__class__(super().__rshift__(other))

    def __rrshift__(self, other: Union[int, torch.Tensor]) -> Self:
        return self.__class__(super().__rrshift__(other))

    def __neg__(self) -> Self:
        return self.__class__(super().__neg__())

    def __pos__(self) -> Self:
        return self.__class__(super().__pos__())

    def __abs__(self) -> Self:
        return self.__class__(super().__abs__())

    def __invert__(self) -> Self:
        return self.__class__(super().__invert__())

    def clone(self, memory_format: Optional[torch.memory_format] = None) -> Self:
        return self.__class__(super().clone(memory_format=memory_format))

    def detach(self) -> Self:
        return self.__class__(super().detach())

    def int(self) -> Self:
        return self.__class__(super().int())

    def float(self) -> Self:
        return self.__class__(super().float())

    def long(self) -> Self:
        return self.__class__(super().long())

    def bool(self) -> Self:
        return self.__class__(super().bool())

    def half(self) -> Self:
        return self.__class__(super().half())

    def double(self) -> Self:
        return self.__class__(super().double())

    def short(self) -> Self:
        return self.__class__(super().short())

    def byte(self) -> Self:
        return self.__class__(super().byte())

    def char(self) -> Self:
        return self.__class__(super().char())

    def requires_grad_(self, requires_grad: bool = True) -> Self:
        super().requires_grad_(requires_grad)
        return self

    def cpu(self) -> Self:
        return self.__class__(super().cpu())

    def cuda(
        self,
        device: Optional[Union[int, str, torch.device]] = None,
        non_blocking: bool = False,
    ) -> Self:
        return self.__class__(super().cuda(device=device, non_blocking=non_blocking))

    def to(
        self,
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = False,
        copy: bool = False,
        *,
        memory_format: Optional[torch.memory_format] = None,
    ) -> Self:
        return self.__class__(
            super().to(
                dtype=dtype,
                non_blocking=non_blocking,
                copy=copy,
                memory_format=memory_format,
            )
        )

    def reshape(self, *shape) -> Self:
        return self.__class__(super().reshape(*shape))

    def view(self, *shape) -> Self:
        return self.__class__(super().view(*shape))

    def transpose(self, dim0: int, dim1: int) -> Self:
        return self.__class__(super().transpose(dim0, dim1))

    def permute(self, *dims) -> Self:
        return self.__class__(super().permute(*dims))

    def squeeze(self, dim: Optional[int] = None) -> Self:
        return self.__class__(super().squeeze(dim=dim))

    def unsqueeze(self, dim: int) -> Self:
        return self.__class__(super().unsqueeze(dim))

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> Self:
        return self.__class__(super().flatten(start_dim, end_dim))

    def expand(self, *sizes) -> Self:
        return self.__class__(super().expand(*sizes))

    def expand_as(self, other: torch.Tensor) -> Self:
        return self.__class__(super().expand_as(other))

    def repeat(self, *sizes) -> Self:
        return self.__class__(super().repeat(*sizes))

    def repeat_interleave(
        self, repeats: Union[int, torch.Tensor], dim: Optional[int] = None
    ) -> Self:
        return self.__class__(super().repeat_interleave(repeats, dim=dim))

    def __getitem__(self, key) -> Self:
        return self.__class__(super().__getitem__(key))

    def select(self, dim: int, index: int) -> Self:
        return self.__class__(super().select(dim, index))

    def index_select(self, dim: int, index: torch.Tensor) -> Self:
        return self.__class__(super().index_select(dim, index))

    def masked_select(self, mask: torch.Tensor) -> Self:
        return self.__class__(super().masked_select(mask))

    def take(self, index: torch.Tensor) -> Self:
        return self.__class__(super().take(index))

    def gather(self, dim: int, index: torch.Tensor) -> Self:
        return self.__class__(super().gather(dim, index))

    def abs(self) -> Self:
        return self.__class__(super().abs())

    def sqrt(self) -> Self:
        return self.__class__(super().sqrt())

    def exp(self) -> Self:
        return self.__class__(super().exp())

    def log(self) -> Self:
        return self.__class__(super().log())

    def sin(self) -> Self:
        return self.__class__(super().sin())

    def cos(self) -> Self:
        return self.__class__(super().cos())

    def tan(self) -> Self:
        return self.__class__(super().tan())

    def tanh(self) -> Self:
        return self.__class__(super().tanh())

    def sigmoid(self) -> Self:
        return self.__class__(super().sigmoid())

    def relu(self) -> Self:
        return self.__class__(super().relu())

    def clamp(self, min: Optional[float] = None, max: Optional[float] = None) -> Self:
        return self.__class__(super().clamp(min=min, max=max))

    def round(self) -> Self:
        return self.__class__(super().round())

    def floor(self) -> Self:
        return self.__class__(super().floor())

    def ceil(self) -> Self:
        return self.__class__(super().ceil())

    def sign(self) -> Self:
        return self.__class__(super().sign())

    def sum(
        self, dim: Optional[Union[int, tuple]] = None, keepdim: bool = False
    ) -> Self:
        return self.__class__(super().sum(dim=dim, keepdim=keepdim))

    def mean(
        self, dim: Optional[Union[int, tuple]] = None, keepdim: bool = False
    ) -> Self:
        return self.__class__(super().mean(dim=dim, keepdim=keepdim))

    def std(
        self, dim: Optional[Union[int, tuple]] = None, keepdim: bool = False
    ) -> Self:
        return self.__class__(super().std(dim=dim, keepdim=keepdim))

    def var(
        self, dim: Optional[Union[int, tuple]] = None, keepdim: bool = False
    ) -> Self:
        return self.__class__(super().var(dim=dim, keepdim=keepdim))

    def min(
        self, dim: Optional[int] = None, keepdim: bool = False
    ) -> Union[Self, tuple[Self, torch.Tensor]]:
        result = super().min(dim=dim, keepdim=keepdim)
        if isinstance(result, tuple):
            return (self.__class__(result[0]), result[1])
        return self.__class__(result)

    def max(
        self, dim: Optional[int] = None, keepdim: bool = False
    ) -> Union[Self, tuple[Self, torch.Tensor]]:
        result = super().max(dim=dim, keepdim=keepdim)
        if isinstance(result, tuple):
            return (self.__class__(result[0]), result[1])
        return self.__class__(result)

    def argmin(self, dim: Optional[int] = None, keepdim: bool = False) -> Self:
        return self.__class__(super().argmin(dim=dim, keepdim=keepdim))

    def argmax(self, dim: Optional[int] = None, keepdim: bool = False) -> Self:
        return self.__class__(super().argmax(dim=dim, keepdim=keepdim))
