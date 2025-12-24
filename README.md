/*
Federated Learning Privacy System
================================

Design and Implementation of a Privacy-Preserving Federated Learning System
under Non-IID Data

1. Project Overview
-------------------
This project implements an end-to-end federated learning (FL) system with
client-level differential privacy (DP), designed to study the privacy–utility
trade-off under realistic conditions:

- Non-IID client data distributions
- Heterogeneous client dataset sizes
- Client-side DP-SGD with Opacus
- Centralized orchestration with federated aggregation

The system is experiment-driven and produces quantitative privacy guarantees (ε)
alongside global model performance metrics.

2. System Architecture
----------------------
Workflow:
1. Raw time-series energy data is preprocessed into a supervised learning task
2. Dataset is split into multiple non-IID client datasets
3. Clients train locally using DP-SGD
4. Server aggregates updates using FedAvg
5. Privacy loss (ε) and global MSE are tracked
6. DP noise sweeps quantify privacy–utility trade-offs

Core components:
- FL Client (local DP training)
- FL Server (model distribution and aggregation)
- Experiment runner
- DP sweep engine
- Evaluation layer

3. Methods Used
---------------
Federated Learning:
- Algorithm: Federated Averaging (FedAvg)
- Rounds: 5
- Local epochs: 1
- Batch size: 128
- Model: Lightweight MLP regressor

Non-IID Data:
- Temporal partitioning
- Unequal client dataset sizes
- No centralized data sharing

Differential Privacy:
- Mechanism: DP-SGD (Opacus)
- Clipping norm: 1.0
- Noise multipliers: 0.3, 0.6, 1.0, 2.0
- Privacy accounting: RDP to ε
- Metrics: mean ε and max ε across clients

4. Experiments
--------------
- Baseline DP-Federated training
- Controlled privacy–utility sweep
- Full FL run per configuration
- Results serialized to JSON

5. Results
----------
Noise 0.3:
- ε_mean ≈ 20.9
- ε_max ≈ 35.3
- MSE ≈ 0.657

Noise 0.6:
- ε_mean ≈ 2.3
- ε_max ≈ 7.9
- MSE ≈ 0.660

Noise 1.0:
- ε_mean ≈ 0.7
- ε_max ≈ 2.6
- MSE ≈ 0.662

Noise 2.0:
- ε_mean ≈ 0.3
- ε_max ≈ 0.9
- MSE ≈ 0.666

6. Analysis
-----------
- Privacy improves rapidly with moderate noise
- Diminishing accuracy returns at high ε
- Convex privacy–utility trade-off observed
- Smaller clients experience higher privacy loss

Noise ≈ 0.6 represents an industry-optimal trade-off.

7. Key Takeaways
----------------
- Quantified privacy–utility behavior in DP-FL
- Realistic non-IID client simulation
- Formal privacy accounting
- Reproducible, research-grade experimentation

8. Reproducibility
------------------
Run experiment:
python scripts/run_experiment.py

Run sweep:
python scripts/run_sweep.py

Generate plot:
python scripts/plot_privacy_utility.py

9. Future Work
--------------
- Centralized vs FL vs DP-FL comparison
- Client dropout simulation
- Secure aggregation
- Fairness-aware privacy metrics
*/
