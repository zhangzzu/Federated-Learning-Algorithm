# Federated Learning Averaging Algorithm

This is a simple implementation of **Federated Learning (FL)** . The bare FL model (without DP) is the reproduction of the paper [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629). 

## Requirements
- torch 1.9.19
- numpy 1.20.3

## Files
> main.py: start this project

> center.py: server of the federated learning

> client.py: client of the federated learning

> datatset.py: data of mnist

> algorithm.py: fedavg algorithm

> models.py: all kind of models

## Usag
1. Run ```python main.py```

## Reference
[1] McMahan, Brendan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-Efficient Learning of Deep Networks from Decentralized Data. In *Proc. Artificial Intelligence and Statistics (AISTATS)*, 2017.

