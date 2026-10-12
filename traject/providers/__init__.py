from .base import Provider


def get_provider(name: str, model: str | None = None) -> Provider:
    """name: 'anthropic' | 'openai' | 'mock'. Imports lazily so one missing SDK doesn't break the other."""
    if name == "anthropic":
        from .anthropic_ import AnthropicProvider

        return AnthropicProvider(model=model or AnthropicProvider.DEFAULT_MODEL)
    if name == "openai":
        from .openai_ import OpenAIProvider

        return OpenAIProvider(model=model or OpenAIProvider.DEFAULT_MODEL)
    if name == "mock":
        from .mock import MockProvider

        return MockProvider()
    raise ValueError(f"unknown provider {name!r}; expected anthropic | openai | mock")


__all__ = ["Provider", "get_provider"]
