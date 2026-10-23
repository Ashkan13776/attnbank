"""Import every category so methods register themselves."""
from . import compress, coupling, exact, factorize, interaction, recurrent, restrict  # noqa: F401

__all__ = ["compress", "coupling", "exact", "factorize", "interaction",
           "recurrent", "restrict"]
