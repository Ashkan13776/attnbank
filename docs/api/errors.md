# Errors

Every error carries three things: *what* went wrong, *where* — which argument, which
method — and *what to try instead*.

Each class subclasses both `AttnBankError` and the built-in exception a caller would
naturally catch, so `except KeyError` around `ab.build(...)` and `except ValueError`
around a forward pass keep working.

::: attnbank.errors

## Backend errors

::: attnbank.backends.BackendNotSupportedError
::: attnbank.backends.BackendFallbackWarning
