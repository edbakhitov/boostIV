## Intro

Code for  

**Causal Gradient Boosting: Boosted Instrumental Variable Regression**  
Edvard Bakhitov and Amandeep Singh  
https://arxiv.org/abs/2101.06078

## Installation
Before running any script you need to install the boostIV package by running

```python setup.py .``` 

The main dependencies are numpy, sklearn and numdifftools which will be automatically downloaded
by running the ```setup.py``` command above. 

## boostIV package
This will install the package ```boostIV``` on your local python environment, which contains boostIV and post-boostIV estimators
from the paper.

## Third party material
The folder ```DeepGMM``` contains the source files from the git repo https://github.com/CausalML/DeepGMM 
that contains the code from the work of Bennet et al. "Deep Generalized Method of Moments for Instrumental Variable Analysis".

The code in ```monte_carlo/KIV.py``` is based on the Matlab codes from the git repo https://github.com/r4hu1-5in9h/KIV 
for the paper by Singh et al. "Kernel Instrumental Variable Regression".
