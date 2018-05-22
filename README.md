# 02460-NMF-with-GP-priors
This repository contains the final product from DTU course 02460 -
advanced machine learning, where we worked on Non-negative matrix factorization
with Gaussian Process priors for raman spectroscopy.

The main hand-in of the project was the article included in ```article/NMF-GPP-raman.pdf```

## Code
The code for the project is divided into seperate files.

* `core.py` - Main file containing the code for the implementation of NMF-GPP, LS-NMF and the
hamiltonian sampling scheme.

* `raman_plots.py` - Script used to generate the raman spectra plots of the article

* `sampling.py` - Generates a single monte carlo chain (modify for multiple)
and dumps it into the `/chains/` folder.

* `sampling_plots.py` - Script used to generate the sampling plots of the article
based on the generated chain(s) from `sampling.py`
hamiltonian sampling scheme

## Setup
Run the following commands to setup the conda environment.

```
conda env create -f environment.yml
source activate gpp-nmf
```

While running the `sampling.py` script, we encountered a "chains not unique" bug.
From search on the internet, we found that the current (only) way to solve the problem
was to change the source code according to the [this comment](https://github.com/pymc-devs/pymc3/issues/2856#issuecomment-366039215).
This source code manipulation is required to run the sampling script.

