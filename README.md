# BIASD
Bayesian Inference for the Analysis of Sub-temporal-resolution Data
Version 0.2

## Manuscript
Increasing the time resolution of single-molecule experiments with Bayesian inference.
Colin D Kinz-Thompson, Ruben L Gonzalez Jr.,
bioRxiv 099648; doi: https://doi.org/10.1101/099648
https://www.biorxiv.org/content/early/2017/05/26/099648

## Notes
See the [documentation](http://biasd.readthedocs.io/) or open `Documentation.html` for more information.


## Development (Testing)
```
pip install -e ".[test]"
pytest --verbose
```
If something failed, check it out with, e.g. `python tests/test_speed.py`