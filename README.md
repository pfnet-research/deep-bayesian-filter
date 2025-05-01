# Deep Bayesian Filter for Bayes-faithful Data Assimilation

## Setup using Python virtualenv

The script is tested with Python version 3.10 and the libraries listed in requirementes.txt, 
and a supported GPU.

```
$ python3.10 -m venv ./venv-dbf
$ . ./venv-dbf/bin/activate
$ pip3 install -r requirements.txt
```

## Run experiments

Double pendulum
```
$ bash ./run_train_double_pendulum_DDP_obsnoise01_dt003_1.sh
```

Lorenz96, direct observation
```
$ bash ./run_DDP_obsnoise1_dt003_cond1.sh
```

Lorenz96, non-linear observation
```
$ bash ./run_DDP_obsnoise1_dt003_cond1.sh
```

## Acknowledgement

This research work was financially supported by the Ministry of Internal Affairs and Communications of Japan with a scheme of  "Research and development of advanced technologies for a user-adaptive remote sensing data platform" (JPMI00316).

