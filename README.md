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

## Install dependencies
```bash
pip install -r requirements.txt
```

## Save installed dependencies
```bash
pip freeze > requirements.txt
```

## Run experiments
```bash
python -m experiments.run_adam_experiment
python -m experiments.run_stn_experiment
python -m experiments.run_delta_stn_experiment
```

## Make plots
After experiments of specific models are complete we can plot them with the command below.
Possible models are "adam", "stn" and "delta_stn".
```bash
python3 scripts/make_plots.py <model_name>
```