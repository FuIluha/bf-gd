"""Interactive LDPC decoder diagnostics explorer."""


def create_app(*args, **kwargs):
    """Import the web layer lazily so the numerical core stays lightweight."""
    from .app import create_app as app_factory

    return app_factory(*args, **kwargs)


__all__ = ["create_app"]
