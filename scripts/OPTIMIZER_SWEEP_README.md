# Optimizer Sweep Training Scripts

This directory contains scripts for running comprehensive optimizer sweeps on the neural signal processing model.

## Overview

The optimizer sweep includes state-of-the-art and novel optimizers:

### Optimizers Included

1. **AdamW** - Decoupled weight decay regularization (SOTA baseline)
2. **Lion** - Evolved sign momentum optimizer (Google 2023)
3. **Sophia** - Second-order clipped stochastic optimization (2023)
4. **AdEMAMix** - Exponential moving average mixture (2024)
5. **Prodigy** - Adaptive learning rate with auto-tuning
6. **Adan** - Adaptive Nesterov momentum (2022)
7. **LAMB** - Layer-wise adaptive moments (large batch training)
8. **Novograd** - Normalized gradient adaptive (NVIDIA)
9. **AdamW + SWA** - Stochastic Weight Averaging
10. **AdamW + Lookahead** - Lookahead optimizer wrapper

### Learning Rate Schedulers

- Cosine Annealing
- Cosine Annealing with Warm Restarts
- OneCycle
- Polynomial Decay
- Exponential Decay
- Linear Decay

### Hyperparameter Search Space

- Learning rates: Multiple values per optimizer (typically 3-5 values)
- Weight decay: [1e-5, 1e-4, 1e-3]
- Beta2 (Adam-style optimizers): [0.98, 0.99, 0.999]
- Optimizer-specific parameters (e.g., Lion's momentum, Sophia's rho, etc.)

## Files

- `train_optimizer_sweep.py` - Main training script with optimizer sweep
- `analyze_sweep_results.py` - Analysis and visualization script
- `OPTIMIZER_SWEEP_README.md` - This file

## Usage

### 1. Run the Sweep

```bash
cd /home/iseanbhanot/Neural-Signal-Processing/scripts
python train_optimizer_sweep.py
```

This will:
- Train models with all optimizer/scheduler combinations
- Save models and results to `/home/iseanbhanot/optimizer_sweep_results/`
- Each run saves:
  - `best_model.pt` - Model with best validation CER
  - `final_model.pt` - Model at end of training
  - `config.json` - Full configuration
  - `results.pkl` - Training metrics (loss, CER, learning rates)

### 2. Quick Test Run

To test with just a few configurations:

```python
# Edit train_optimizer_sweep.py at the bottom
run_optimizer_sweep(
    base_args=base_args,
    optimizer_configs=optimizer_configs[:3],  # Just first 3 optimizers
    scheduler_configs=scheduler_configs[:2],  # Just first 2 schedulers
    max_configs=5,  # Limit total configs
)
```

### 3. Analyze Results

```bash
python analyze_sweep_results.py --sweep-dir /home/iseanbhanot/optimizer_sweep_results
```

This generates:
- `rankings.csv` - Ranked table of all configurations
- `report.txt` - Summary statistics and best configs
- `training_curves.png` - Training progress plots
- `optimizer_comparison.png` - Box plots by optimizer
- `scheduler_comparison.png` - Box plots by scheduler
- `hyperparameter_heatmap.png` - LR vs weight decay heatmap

### 4. Load and Use Best Model

```python
import torch
import pickle
import json

# Load best run's model
run_dir = '/home/iseanbhanot/optimizer_sweep_results/AdamW_lr0.002_wd0.0001_b20.999_cosine_run0'

# Load config
with open(f'{run_dir}/config.json', 'r') as f:
    config = json.load(f)

# Load model
from neural_decoder.model import GRUDecoder

model = GRUDecoder(
    neural_dim=config['base_args']['nInputFeatures'],
    n_classes=config['base_args']['nClasses'],
    hidden_dim=config['base_args']['nUnits'],
    layer_dim=config['base_args']['nLayers'],
    nDays=24,  # Adjust as needed
    dropout=config['base_args']['dropout'],
    device='cuda',
    strideLen=config['base_args']['strideLen'],
    kernelLen=config['base_args']['kernelLen'],
    gaussianSmoothWidth=config['base_args']['gaussianSmoothWidth'],
    bidirectional=config['base_args']['bidirectional'],
).to('cuda')

model.load_state_dict(torch.load(f'{run_dir}/best_model.pt'))
model.eval()
```

## Configuration

### Customize Base Arguments

Edit `base_args` in `train_optimizer_sweep.py`:

```python
base_args = {
    'sweep_output_dir': '/path/to/output',
    'datasetPath': '/path/to/dataset.pkl',
    'batchSize': 32,
    'nBatch': 10000,  # Number of training batches
    'eval_every': 100,  # Evaluation frequency
    # ... other params
}
```

### Add Custom Optimizers

Add to `get_optimizer_configs()`:

```python
configs.append({
    'name': 'MyOptimizer_lr0.001',
    'optimizer': 'MyOptimizer',
    'lr': 0.001,
    # ... other hyperparameters
})
```

Then implement in `get_optimizer()`:

```python
elif optimizer_name == 'MyOptimizer':
    return MyOptimizer(
        model.parameters(),
        lr=config['lr'],
        # ... other params
    )
```

### Add Custom Schedulers

Add to `get_scheduler_configs()` and implement in `get_scheduler()`.

## Output Structure

```
optimizer_sweep_results/
├── AdamW_lr0.002_wd0.0001_b20.999_cosine_run0/
│   ├── best_model.pt
│   ├── final_model.pt
│   ├── config.json
│   └── results.pkl
├── Lion_lr0.0003_wd0.001_b20.99_onecycle_run1/
│   ├── best_model.pt
│   ├── final_model.pt
│   ├── config.json
│   └── results.pkl
├── ... (more runs)
├── summary.pkl
└── analysis/
    ├── rankings.csv
    ├── report.txt
    ├── training_curves.png
    ├── optimizer_comparison.png
    ├── scheduler_comparison.png
    └── hyperparameter_heatmap.png
```

## Expected Runtime

With default settings:
- **Single run**: ~1-2 hours (10,000 batches on GPU)
- **Full sweep**: Varies based on number of configs
  - ~100 configurations × 1.5 hours = ~150 hours
  - Can run in parallel on multiple GPUs

## Tips

1. **Start small**: Test with `max_configs=5` first
2. **Use multiple GPUs**: Modify script to distribute configs across GPUs
3. **Monitor progress**: Check output logs for issues
4. **Disk space**: Each run ~100-500MB, plan accordingly
5. **Resume failed runs**: Script skips existing directories

## Advanced Features

### Gradient Clipping

Enabled by default (`grad_clip: 1.0`). Adjust in `base_args`.

### Stochastic Weight Averaging (SWA)

Enabled for specific configs (e.g., `AdamW_SWA`). Starts at 75% of training.

### Lookahead Wrapper

Enabled for specific configs. Uses k=5, alpha=0.5.

### Early Stopping

Not implemented by default. Add to `train_single_config()` if desired.

## Troubleshooting

### Out of Memory
- Reduce `batchSize` in `base_args`
- Reduce `nUnits` or `nLayers`

### Missing Dependencies
- Script auto-installs optimizer packages
- Manually install: `pip install lion-pytorch sophia-optimizer prodigyopt`

### Slow Training
- Reduce `nBatch` for faster testing
- Use smaller model architecture
- Check GPU utilization with `nvidia-smi`

## Citation

If you use these optimizers, please cite the respective papers:

- **Lion**: Chen et al., "Symbolic Discovery of Optimization Algorithms" (2023)
- **Sophia**: Liu et al., "Sophia: A Scalable Stochastic Second-order Optimizer" (2023)
- **AdEMAMix**: Pagliardini et al., "The AdEMAMix Optimizer" (2024)
- **Prodigy**: Mishchenko & Defazio, "Prodigy: An Expeditiously Adaptive Parameter-Free Learner" (2023)
- **Adan**: Xie et al., "Adan: Adaptive Nesterov Momentum Algorithm" (2022)

## References

- AdamW: https://arxiv.org/abs/1711.05101
- Lion: https://arxiv.org/abs/2302.06675
- Sophia: https://arxiv.org/abs/2305.14342
- Prodigy: https://arxiv.org/abs/2306.06101
- Adan: https://arxiv.org/abs/2208.06677
