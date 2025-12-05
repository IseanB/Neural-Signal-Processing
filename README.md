# Neural Decoder Optimization: A Comparative Study of Modern Optimizers

_**By: Isean Bhanot**_

This project investigates the impact of modern optimization algorithms on the performance of a Gated Recurrent Unit (GRU) based neural decoder for speech synthesis from intracortical neural signals. I conducted a comprehensive hyperparameter sweep comparing industry-standard optimizers (AdamW) against recently proposed novel optimizers (Lion, Sophia, Prodigy). 1results demonstrate that novel adaptive optimizers, specifically Prodigy and Lion, consistently outperform the AdamW baseline. Achieving a CER of 0.2033, the Prodigy-OneCycle configuration yielded an 9.4% relative improvement compared to the best AdamW baseline (CER 0.2245). This suggests that exploring the optimizer landscape is a highleverage pathway for improving neural decoding performance without increasing model complexity.

<img width="5154" height="3375" alt="comprehensive_dashboard" src="https://github.com/user-attachments/assets/25222844-7264-498e-9908-90f4672d041b" />




## Requirements
- python >= 3.9

## Installation

pip install -e .

## How to run

1. Convert the speech BCI dataset using [formatCompetitionData.ipynb](./notebooks/formatCompetitionData.ipynb)
2. Prepare Dataset: `python scripts/prepare_data_lowmem.py`
3. Run Training Sweep: `python scripts/train_optimizer_sweep.py`
4. Visualize: `python scripts/visualize_sweep_results.py`

## Baseline GRU Implementation: [Neural Sequence Decoder](https://github.com/fwillett/speechBCI/tree/main/NeuralDecoder)
