"""Domain errors with stable CLI exit semantics."""


class ResearchTreeError(Exception):
    """Base error for expected user-facing failures."""


class NotFoundError(ResearchTreeError):
    """A requested graph object does not exist."""


class ValidationError(ResearchTreeError):
    """Persisted or supplied graph data is invalid."""


class ProviderError(ResearchTreeError):
    """A model provider request failed (transport, HTTP, or configuration)."""


class ModelOutputError(ProviderError):
    """A provider returned content that failed schema validation.

    Subclasses ProviderError so existing handlers that treat bad model output as a
    provider-side failure keep working, but the CLI maps it to exit code 5
    (validation) so agents can distinguish "retry a different model" from
    "fix the credential or reach the provider".
    """


class ConfigurationError(ResearchTreeError):
    """Required configuration or credentials are missing."""
