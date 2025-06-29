from enum import Enum


class OptimizerClass(Enum):
    """Formal taxonomy of model optimisation behaviours.

    The categories are derived from the theoretical agenda:

    * PROXY_ALIGNED – the model runs optimisation but on a surrogate of the
      outer/training loss (no misalignment but still an inner optimiser).
    * MISGENERALISED – the model generalises its objective incorrectly when
      transferred to out-of-distribution settings.
    * DECEPTIVE – the model has learned to represent an objective aligned with
      training *behaviour* during training yet plans to pursue a different one
      at deployment (it is strategically deceptive).
    * NON_OPTIMISING – heuristics or feed-forward policies that are goal
      directed but do not internally perform optimisation.
    """

    PROXY_ALIGNED = "proxy_aligned"
    MISGENERALISED = "misgeneralised"
    DECEPTIVE = "deceptive"
    NON_OPTIMISING = "non_optimising"

    def __str__(self) -> str:  # pragma: no cover
        return self.value 