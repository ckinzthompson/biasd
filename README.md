# BIASD
Bayesian Inference for the Analysis of Sub-temporal-resolution Data
- Version 0.2 (July 2024)
- Version 0.1 (2017)

[![Documentation Status](https://readthedocs.org/projects/biasd/badge/?version=main)](https://biasd.readthedocs.io/en/main/?badge=main) See the [documentation](http://biasd.readthedocs.io/) or open `Documentation.html` for more information.


## Quick start
Install straight from Github:
``` sh
pip install git+https://github.com/ckinzthompson/biasd.git
```

Try running a speed test:
``` python
import biasd as b
b.likelihood.use_cuda_ll()
t,y = b.likelihood.test_speed(10,1000)
```

## Development (Testing)
``` sh
pip install -e ".[test]"
pytest
```

## Manuscript
Colin D Kinz-Thompson, Ruben L Gonzalez Jr. 2018. Increasing the time resolution of single-molecule experiments with Bayesian inference. Biophysical Journal. 114(2), 289-300.

[Link (Pubmed)](https://pubmed.ncbi.nlm.nih.gov/29401427/)

[Link (bioRxiv)](https://www.biorxiv.org/content/early/2017/05/26/099648)
