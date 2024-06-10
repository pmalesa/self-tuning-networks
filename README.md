# Self-Tuning Networks
Authors: Szczepaniak Katarzyna & Malesa Piotr

## Virtual environment
### Setup
```bash
python3 -m venv venv
```

### Activation
```bash
source venv/bin/activate
```

### Deactivation
```bash
source deactivate
```

## Install dependencies
```bash
pip install -r requirements.txt
```

## Save installed dependencies
```bash
pip freeze > requirements.txt
```

## Run experiments
### Neural network model with Adam optimizer experiment
```bash
python -m experiments.run_adam_experiment
```
### STN model experiment
```bash
python -m experiments.run_stn_experiment
```
### Delta-STN model experiment
```bash
python -m experiments.run_stn_experiment --delta_stn
```

## Make plots
After experiments of specific models are complete we can plot them with the command below.
Possible models are "adam", "stn" and "delta_stn".
```bash
python3 scripts/make_plots.py <model_name>
```