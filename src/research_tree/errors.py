"""Domain errors with stable CLI exit semantics."""


class ResearchTreeError(Exception):
    """Base error for expected user-facing failures."""


class NotFoundError(ResearchTreeError):
    """A requested graph object does not exist."""


class ValidationError(ResearchTreeError):
    """Persisted or supplied graph data is invalid."""


class ProviderError(ResearchTreeError):
    """A model provider request failed."""


class ConfigurationError(ResearchTreeError):
    """Required configuration or credentials are missing."""
