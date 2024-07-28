# BIASD: Bayesian Inference for the Analysis of Sub-temporal-resolution Data

[![Documentation Status](https://readthedocs.org/projects/biasd/badge/?version=main)](https://biasd.readthedocs.io/en/main/?badge=main)

## Quick start
Install straight from Github:
``` sh
pip install git+https://github.com/ckinzthompson/biasd.git
```

Try running a speed test:
``` python
import biasd as b
b.likelihood.use_python_ll()
t,y = b.likelihood.test_speed(10,1000)
```

## Documentation
See the [documentation](http://biasd.readthedocs.io/) or open `Documentation.html` for more information.

## Manuscripts
#### Original Paper
* Kinz-Thompson, CD, Gonzalez Jr., RL 2018. _Increasing the time resolution of single-molecule experiments with Bayesian inference_. Biophysical Journal. 114(2), 289-300. [(Biophysical Journal)](https://www.cell.com/biophysj/fulltext/S0006-3495(17)34973-1),  [(Pubmed)](https://pubmed.ncbi.nlm.nih.gov/29401427/),  [(bioRxiv)](https://www.biorxiv.org/content/early/2017/05/26/099648).

#### Applications
* Ray, K., Kinz-Thompson, C.D., Fei, J., Wang, B., Lin, Q. and Gonzalez Jr., R.L. 2023. _Entropic control of the free-energy landscape of an archetypal biomolecular machine_. Proc. Natl. Acad. Sci. U.S.A. 120:e2220591120. [(PNAS)](https://pnas.org/doi/10.1073/pnas.2220591120),  [(Pubmed)](https://pubmed.ncbi.nlm.nih.gov/37186858),  [(bioRxiv)](https://www.biorxiv.org/content/10.1101/2022.10.03.510626v2).

## Development (Testing)
``` sh
pip install -e ".[test]"
pytest
```

## Updates
* Version 0.2 (July 2024): Updates fix broken libraries and improve clarity.
* Version 0.1 (2017): Original used in paper

