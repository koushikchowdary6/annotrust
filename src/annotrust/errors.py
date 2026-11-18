"""User-facing errors."""


class AnnotrustError(Exception):
    """The input or configuration is not usable.

    The CLI prints these and exits 2. They are not annotator findings.
    """
