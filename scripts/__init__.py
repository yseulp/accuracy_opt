import os

_enable_jax_x64 = os.environ.get("ACCURACY_OPT_ENABLE_JAX_X64", "true").lower()

if _enable_jax_x64 in {"1", "true", "yes", "on"}:
    import jax

    jax.config.update("jax_enable_x64", True)
