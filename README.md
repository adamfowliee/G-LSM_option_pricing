# G-LSM American Option Pricing

This repository contains my work on the Gradient-Enhanced Least Squares Monte Carlo (G-LSM) method for pricing high-dimensional American (Bermudan) options.

## Structure

- `PaperReview_Article.pdf`  
  My short report explaining the method, its context, and experimental results.

- `glsm-american/`  
  Original MATLAB code from the paper (reference implementation).

- `python_code/`  
  My Python implementation and experiments.

## Python notebooks

- `understanding_functions.ipynb`  
  Walkthrough of key functions used in the implementation, with explanations.

- `calibrate_model.ipynb`  
  Runs the model using real market data.  
  Compares G-LSM prices with Black-Scholes and binomial tree benchmarks.

- `test_results.ipynb`  
  Generates plots used in the report.  
  Includes experiments across different moneyness levels.

## Notes

- The focus is on geometric basket options in multiple dimensions.
- Experiments include comparisons to analytical and tree-based benchmarks.
