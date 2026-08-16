# GA-BPNN for Concrete Strength Prediction

A machine-learning study of **genetic-algorithm-tuned neural regression** for predicting the compressive strength of alkali-residue-based foamed concrete (AR-FC).

The project compares a standard backpropagation neural network with a GA-tuned BPNN that searches batch size, learning rate, training epochs, hidden-layer widths, and dropout. In the recorded experiment, GA tuning increased the reported R² from **0.8744 to 0.9307** while reducing RMSE from **0.3465 to 0.2574**.

## Key results

| Model | MAE | RMSE | MAPE | R² |
|---|---:|---:|---:|---:|
| Default BPNN | 0.2614 | 0.3465 | 0.2704 | 0.8744 |
| **GA-BPNN** | **0.2275** | **0.2574** | **0.2513** | **0.9307** |

Relative to the default network, the GA-tuned model reduced RMSE by about **25.7%** and MAE by about **13.0%**, while increasing R² by about **6.4%**.

## Problem

AR-FC is a lightweight cementitious material incorporating industrial alkali residue. Its compressive strength depends on coupled material and process variables, making it a useful engineering-regression problem for data-driven modeling.

The model uses four input features:

- OPC ratio
- GGBS ratio
- wet density
- water-cement ratio

and predicts compressive strength as a continuous response.

## Models

### Default BPNN

The baseline is a feed-forward neural network with:

- two hidden layers
- PReLU activations
- dropout regularization
- Adam optimization
- standardized input and target variables

Recorded baseline architecture: **32 → 32** hidden units.

### GA-BPNN

A genetic algorithm searches six neural-network/training hyperparameters:

- batch size
- learning rate
- epochs
- first hidden-layer width
- second hidden-layer width
- dropout probability

The recorded best configuration was:

| Hyperparameter | Value |
|---|---:|
| Batch size | 103 |
| Learning rate | 0.004765 |
| Epochs | 123 |
| Hidden layer 1 | 18 |
| Hidden layer 2 | 128 |
| Dropout | 0.1 |

This turns network configuration into an explicit optimization problem rather than relying on a single manually selected architecture.

## Repository structure

- `main_default.py` — original baseline BPNN experiment
- `main_ga.py` — original GA hyperparameter-search experiment
- `run_default.py` — portable entry point for the baseline experiment
- `run_ga.py` — portable entry point for GA-BPNN optimization
- `data_io.py` — shared portable dataset loader
- `outputs_default/` — recorded baseline outputs

The original experiment scripts are kept intact as the historical implementation used for the reported results. The portable entry points provide a cleaner way to run the same workflow on another machine.

## Installation

```bash
pip install -r requirements.txt
```

## Dataset format

Provide a headerless CSV with five columns:

```text
x1,x2,x3,x4,y
```

where the first four columns are the input variables and the fifth column is compressive strength.

By default the portable runners look for:

```text
data.csv
```

You can also point to another file without editing source code:

```bash
export CONCRETE_DATA=/path/to/data.csv
```

On Windows PowerShell:

```powershell
$env:CONCRETE_DATA="C:\path\to\data.csv"
```

## Run the baseline

```bash
python run_default.py
```

## Run GA-BPNN

```bash
python run_ga.py
```

The GA workflow performs hyperparameter search, trains the selected network, evaluates predictions, and writes model artifacts and visualizations under `outputs_ga/`.

## Evaluation

The experiment reports:

- mean absolute error (MAE)
- root mean squared error (RMSE)
- mean absolute percentage error (MAPE)
- coefficient of determination (R²)

The stored results come from the fixed 80/20 experimental split used by the original scripts. The GA uses that holdout performance to guide hyperparameter selection, so the reported metrics should be interpreted as the performance of the recorded model-selection experiment rather than as a separate external benchmark.

## Engineering takeaway

The project demonstrates how evolutionary search can systematically tune a neural regression model for a materials-engineering problem. The strongest result is not only the higher R², but the simultaneous reduction in absolute and squared prediction error after GA-based hyperparameter optimization.

## References

1. Wang, Z., Liu, S., Wu, K., Huang, L., Wang, J. (2023). *Study on the mechanical performance of alkali residue-based lightweight soil*. Construction and Building Materials, 384, 131353.
2. Elbaz, K. (2021). Flowchart of generalized structure for GA model.

## Contributors

- Yuhao Zhang
- Daniel Ye
