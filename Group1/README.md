# Deep-Q-Network-Electric-Cars
This is the source code accompanying the report entitled "Optimizing Electric Vehicle Energy Trading with Deep Reinforcement Learning"

## Requirements
Install the requirements by running:
```sh
pip install -r requirements.txt
```

## Algorithms
We implemented 5 algorithms in total, 2 baselines (EMA and BLSH) and 3 Reinforcement learning algorithms (Q-learning, DQN and PG). Change data/validate.xlsx to the desired dataset. 

To run EMA: 
```sh
python3 main.py ema data/validate.xlsx
```

To run BLSH:
```sh
python3 main.py blsh data/validate.xlsx
```

To run Q-learning:
```sh
python3 main.py qlearning data/validate.xlsx
```

To run DQN:
```sh
python3 main.py DQN data/validate.xlsx
```

To run PG:
```sh
python3 main.py PG data/validate.xlsx
```
## Training
These command are for retraining the models:

For Q-learning:
```sh
python3 qlearning_train.py
```

For DQN:
```sh
python3 DQN_train.py
```

For PG
```sh
python3 pg_lstm.py
```



