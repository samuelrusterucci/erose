<div align="center">
  <picture>
    <source
    srcset="./logos/DarkkMode.png"
    media="(prefers-color-scheme: dark)"
    width="100%" height="100%"
    />
    <img
    src="./logos/LighttMode.png"
    width="100%" height="100%"
    />
  </picture>
    
  <h3 align="center">Enhancing Rapidly Overdensities of Sources with Ease</h3>
</div>

[![PyPI](https://img.shields.io/pypi/v/erose.svg)](https://pypi.python.org/pypi/erose)

## Overview

`erose` is a Python package that automatically searches for dwarf galaxies in stellar catalogues. It requires minimal inputs, namely:

- stellar positions
- stellar magnitudes (and their uncertainties)
- an empirical or theoretical isochrone

and can then process surveys such as UNIONS, DES or Euclid on a standard laptop in less than a day.

A detailed description of the algorithm is provided in [Rusterucci et al. (2026)](). For usage examples, please refer to the [example notebook]().

## Installation

You can install `erose` from PyPI: 
```bash
pip install erose
```
All required dependencies will be installed automatically.

## Citing this work
 
If you use `erose` please consider citing: 

```
@ARTICLE{PhySO_RL_DA,
       author = {{Tenachi}, Wassim and {Ibata}, Rodrigo and {Diakogiannis}, Foivos I.},
        title = "{Deep Symbolic Regression for Physics Guided by Units Constraints: Toward the Automated Discovery of Physical Laws}",
      journal = {ApJ},
         year = 2023,
        month = dec,
       volume = {959},
       number = {2},
          eid = {99},
        pages = {99},
          doi = {10.3847/1538-4357/ad014c},
archivePrefix = {arXiv},
       eprint = {2303.03192},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023ApJ...959...99T},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

