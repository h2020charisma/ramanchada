# ramanchada
A tool for harmonisation of Raman spectral data by characteristic data (CHADA)

# Quick start
Ramanchada and some of its external dependencies are not (yet) available as PyPI or Anaconda packages, so, currently,
the recommended method of installation is directly from the Git repo. Only `pip` is supported:

```
pip install --user -e .
```

As usual, the `--user` flag may be omitted if a system-wide installation is required or appropriate, e.g. when some
type of virtual environment is used: venv, conda, Docker.

`python setup.py install` is **not** supported; however, `setup.py sdist` and `setup.py bdist_wheel` should work.
