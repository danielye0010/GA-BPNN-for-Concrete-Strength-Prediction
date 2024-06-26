# Predictive Analysis of Compressive Strength of Alkali Residue-Based Foamed Concrete

## Overview
This project investigates the use of machine learning techniques, particularly Backpropagation Neural Networks (BPNN) and Genetic Algorithms (GA), to predict the compressive strength of Alkali Residue-Based Foamed Concrete (AR-FC).

## Introduction
Alkali Residue-Based Foamed Concrete (AR-FC) is a lightweight material derived from industrial waste. It has significant potential for reuse in construction, offering benefits such as reduced weight, improved insulation, and cost savings. However, predicting its compressive strength is crucial for practical applications.

## Methodology
### Sample Preparation
1. **Materials**: Ordinary Portland Cement (OPC), Ground Granulated Blast Furnace Slag (GGBS), and Alkali Residue.
2. **Slurry Preparation**: Mix OPC, GGBS, and Alkali Residue with water.
3. **Foam Preparation**: Generate foam and incorporate it into the slurry.
4. **Casting**: Pour the mixture into molds, adjust density as needed.
5. **Curing**: Demold samples after 48 hours, cure at 20°C and 95% RH for 28 days.

### Data Preprocessing
- **Cleaning**: Address outliers, handle missing values, remove duplicates.
- **Standardization**: Apply standard scaler to normalize data.

### Model Training
- **Backpropagation Neural Network (BPNN)**:
  - Input: OPC ratio, GGBS ratio, wet density, water-cement ratio.
  - Architecture: Two hidden layers (32 neurons each), output layer predicts compressive strength.
  - Training: Iterative process using backpropagation, evaluated with MAE, RMSE, MAPE, and R-squared.

- **Genetic Algorithm (GA)**:
  - Optimizes BPNN parameters to avoid local optima.
  - Steps: Initialization, selection, crossover, mutation, evaluation, and termination.

## Results
### Default BPNN Model
- **Metrics**:
  - MAE: 0.2614
  - RMSE: 0.3465
  - MAPE: 0.2704
  - R-squared: 0.8744

### Optimized GA-BPNN Model
- **Optimized Parameters**:
  - Batch Size: 103
  - Learning Rate: 0.004765
  - Epochs: 123
  - Hidden Layer Sizes: 18 and 128 neurons
  - Dropout Probability: 0.1
- **Metrics**:
  - MAE: 0.2275
  - RMSE: 0.2574
  - MAPE: 0.2513
  - R-squared: 0.9307

## Discussion
The integration of Genetic Algorithms significantly improved the BPNN's predictive accuracy. These results demonstrate the potential of AI-based models in predicting the compressive strength of AR-FC, paving the way for more efficient and sustainable construction materials.

## References
1. Wang, Z., Liu, S., Wu, K., Huang, L., Wang, J. (2023). Study on the mechanical performance of alkali residue-based lightweight soil. *Construction and Building Materials*, 384, 131353.
2. Elbaz, K. (2021). Flowchart of generalized structure for GA model. 

# Contributor
Yuhao Zhang
Daniel Ye
