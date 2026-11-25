"""tensorplay.vision — vision components.

Organized into ``datasets``, ``transforms``, ``models``, ``ops`` and ``io``
subpackages plus the ``make_grid``/``save_image`` helpers from ``utils``.
"""

from . import datasets
from . import transforms
from . import models
from . import ops
from . import io as io  # noqa: F401
from .backend import set_backend, get_backend
from .transforms.functional import to_tensor
from .utils import make_grid, save_image

__all__ = [
    "datasets",
    "transforms",
    "models",
    "ops",
    "io",
    "set_backend",
    "get_backend",
    "to_tensor",
    "make_grid",
    "save_image",
]
