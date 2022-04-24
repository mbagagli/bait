# BAIT

[![PyPI version](https://badge.fury.io/py/bait.svg)](https://badge.fury.io/py/bait)


**It**erative **Ba**er picking algorithm.

AUTHOR: _Matteo Bagagli_

VERSION: _2.5.9_

DATE: _04/2022_

----------

The BaIt picking system is a software created around the already famous and widely used Baer-Kradolfer picker (Baer 1987). The proposed seismic picker push forward the already great performance of the Baer-Kradolfer algorithm by adding an iterative picking procedure over the given seismic trace.
It relies on standard libraries like `matplotlib` and `obspy`.

## Installation

From version `2.5.9` the `bait` picker is also stored on PyPI.
Prior of the run of the subsequent code, the user must have installed `conda` or `miniconda`.

```bash
$ conda create -n bait python=3.6  # also valid on higher versions.
$ conda activate bait
$ pip install bait # PyPI
```

## Contributing

The `master` branch will remain the official branch for stable releases (and following updates on PyPI).
If you would like to contribute, please fork the project and checkout to the `DEVELOP` branch.
From that branch (that is the latest up-to date versions) please create a new branch to work on.

The branch name should be representative of what are you actually trying to achieve/improve.
The way they should be named are:

- `develop_FEATURENAME`: improve the current state of the software.
- `bugfix_FEATURENAME`: to correct / fix previous broken/faulty code features or library dependencies
- `document_FEATURENAME`: to be used when updating the docs or the function's docstring.

Please check-out the `CODE_OF_CONDUCT.md` file.

----------
#### References

- Baer, M., and U. Kradolfer. "An automatic phase picker for local and teleseismic events." Bulletin of the Seismological Society of America 77.4 (1987): 1437-1445.

##### Funny Quotes
- Always code as if the guy who ends up maintaining your code will be a violent psycopath who knows where you live (Jhon Woods)

