class ModelUnavailableError(RuntimeError):
    pass


class EvidenceUnavailableError(RuntimeError):
    pass


class ToolExecutionError(RuntimeError):
    pass


class LLMRateLimitError(RuntimeError):
    pass


class LLMProviderError(RuntimeError):
    pass


class LLMConfigurationError(RuntimeError):
    pass


class DatabaseError(RuntimeError):
    pass
