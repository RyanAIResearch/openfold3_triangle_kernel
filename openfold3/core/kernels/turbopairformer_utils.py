import importlib
import importlib.util

import torch


def is_turbopairformer_available() -> bool:
    """Check if turbopairformer is installed and CUDA is available."""
    return (
        importlib.util.find_spec("turbopairformer") is not None
        and torch.cuda.is_available()
    )


def validate_turbopairformer_attention_policy() -> None:
    """Require the package's single-reconstruction policy before host fallback."""
    try:
        policy = importlib.import_module("turbopairformer.policy")
        validate = policy.validate_attention_policy
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "This integration requires TurboPairFormer's mandatory "
            "single-reconstruction attention policy. Run the pinned source with "
            "`pixi run -e openfold3 run-current-turbopairformer <command>`; "
            "the historical 0.1.2 wheel is not compatible with this route."
        ) from exc
    validate()
