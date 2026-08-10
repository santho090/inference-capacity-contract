# Contributing

Contributions must preserve the contract boundaries:

- Keep calculations deterministic and unit-labelled.
- Attach provenance and validity bounds to every derived quantity.
- Treat unsupported architectures and missing evidence explicitly; do not
  silently infer compatibility.
- Do not add private traces, prices, fleet details, credentials, or copied
  vendor/internal recipes.
- Keep the default runtime dependency-free and CPU-only.

Before opening a pull request:

```bash
python -m pip install -e '.[dev]'
python -m ruff check .
python -m mypy
python -m unittest discover -s tests -v
python -m compileall -q src tests
python -m pip wheel --no-deps --wheel-dir /tmp/icc-dist .
```
