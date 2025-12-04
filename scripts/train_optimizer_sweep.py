"""
Training script with comprehensive sweep of SOTA and novel optimizers.

This script includes:
- Classic optimizers: SGD, Adam, AdamW
- Modern optimizers: Lion, Sophia, AdEMAMix, Prodigy, Adan
- Experimental: Shampoo, LARS, LAMB, Novograd
- Learning rate schedules: Cosine, OneCycle, Polynomial, Exponential
- Advanced techniques: Gradient clipping, warmup, weight decay schedules
"""

import sys
import os
from pathlib import Path
import json
import itertools
from typing import Dict, List, Any
import datetime
import pickle

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Set PyTorch memory allocation to avoid fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import numpy as np
from neural_decoder.neural_decoder_trainer import getDatasetLoaders
from neural_decoder.model import GRUDecoder


# ============================================================================
# OPTIMIZER CONFIGURATIONS
# ============================================================================

def get_optimizer_configs() -> List[Dict[str, Any]]:
    """
    Returns a list of optimizer configurations to sweep over.
    Each config includes the optimizer class and its hyperparameters.

    Reduced to ~9 configurations (half of original ~17).
    """
    configs = []

    # ------------------------------------------------------------------------
    # 1. AdamW (SOTA baseline - Decoupled Weight Decay Regularization)
    # 3 configurations (reduced from 6)
    # ------------------------------------------------------------------------
    for lr in [1e-3, 2e-3]:
        configs.append({
            'name': f'AdamW_lr{lr}_wd1e-4',
            'optimizer': 'AdamW',
            'lr': lr,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 1e-4,
            'amsgrad': False,
        })

    # One config with different beta2
    configs.append({
        'name': f'AdamW_lr1e-3_b20.99',
        'optimizer': 'AdamW',
        'lr': 1e-3,
        'betas': (0.9, 0.99),
        'eps': 1e-8,
        'weight_decay': 1e-4,
        'amsgrad': False,
    })

    # ------------------------------------------------------------------------
    # 2. Lion Optimizer (Evolved Sign Momentum - Google 2023)
    # 2 configurations (reduced from 4)
    # ------------------------------------------------------------------------
    for lr in [1e-4, 3e-4]:
        configs.append({
            'name': f'Lion_lr{lr}_wd1e-4',
            'optimizer': 'Lion',
            'lr': lr,
            'betas': (0.9, 0.99),
            'weight_decay': 1e-4,
        })

    # ------------------------------------------------------------------------
    # 3. Sophia (Second-order Clipped Stochastic Optimization - 2023)
    # 2 configurations (kept same)
    # ------------------------------------------------------------------------
    for lr in [1e-4, 3e-4]:
        configs.append({
            'name': f'Sophia_lr{lr}',
            'optimizer': 'Sophia',
            'lr': lr,
            'betas': (0.965, 0.99),
            'rho': 0.04,
            'weight_decay': 1e-4,
        })

    # ------------------------------------------------------------------------
    # 4. Prodigy (Adaptive Learning Rate - Auto LR tuning)
    # 2 configurations (kept same)
    # ------------------------------------------------------------------------
    for d0 in [1e-6, 1e-5]:
        configs.append({
            'name': f'Prodigy_d0{d0}',
            'optimizer': 'Prodigy',
            'lr': 1.0,  # Prodigy adapts LR automatically
            'betas': (0.9, 0.999),
            'beta3': 0.9,
            'd0': d0,
            'weight_decay': 1e-4,
        })

    return configs


def get_scheduler_configs() -> List[Dict[str, Any]]:
    """
    Returns learning rate scheduler configurations.

    Reduced to 3 key schedulers.
    17 optimizers × 3 schedulers = 51 configurations (target: 50)
    """
    return [
        {
            'name': 'cosine',
            'type': 'CosineAnnealingLR',
            'T_max': None,  # Will be set to nBatch
            'eta_min': 1e-6,
        },
        {
            'name': 'onecycle',
            'type': 'OneCycleLR',
            'max_lr': None,  # Will be set to optimizer lr
            'total_steps': None,  # Will be set to nBatch
            'pct_start': 0.3,
            'anneal_strategy': 'cos',
        },
        {
            'name': 'exponential',
            'type': 'ExponentialLR',
            'gamma': 0.9999,
        },
    ]


# ============================================================================
# OPTIMIZER IMPLEMENTATIONS
# ============================================================================

def get_optimizer(model, config):
    """
    Returns the optimizer based on the configuration.
    Installs required packages if needed.
    """
    optimizer_name = config['optimizer']

    if optimizer_name == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            betas=config.get('betas', (0.9, 0.999)),
            eps=config.get('eps', 1e-8),
            weight_decay=config.get('weight_decay', 0),
            amsgrad=config.get('amsgrad', False),
        )

    elif optimizer_name == 'Lion':
        try:
            from lion_pytorch import Lion
        except ImportError:
            print("Installing lion-pytorch...")
            os.system("pip install lion-pytorch")
            from lion_pytorch import Lion

        return Lion(
            model.parameters(),
            lr=config['lr'],
            betas=config.get('betas', (0.9, 0.99)),
            weight_decay=config.get('weight_decay', 0),
        )

    elif optimizer_name == 'Sophia':
        try:
            from sophia import SophiaG
        except ImportError:
            print("Installing sophia-optimizer...")
            os.system("pip install sophia-optimizer")
            from sophia import SophiaG

        return SophiaG(
            model.parameters(),
            lr=config['lr'],
            betas=config.get('betas', (0.965, 0.99)),
            rho=config.get('rho', 0.04),
            weight_decay=config.get('weight_decay', 0),
        )

    elif optimizer_name == 'AdEMAMix':
        try:
            from ademamix import AdEMAMix
        except ImportError:
            print("Installing ademamix-optimizer...")
            os.system("pip install git+https://github.com/nanowell/AdEMAMix-Optimizer-Pytorch.git")
            from ademamix import AdEMAMix

        return AdEMAMix(
            model.parameters(),
            lr=config['lr'],
            betas=config.get('betas', (0.9, 0.999)),
            alpha=config.get('alpha', 5.0),
            weight_decay=config.get('weight_decay', 0),
        )

    elif optimizer_name == 'Prodigy':
        try:
            from prodigyopt import Prodigy
        except ImportError:
            print("Installing prodigyopt...")
            os.system("pip install prodigyopt")
            from prodigyopt import Prodigy

        return Prodigy(
            model.parameters(),
            lr=config.get('lr', 1.0),
            betas=config.get('betas', (0.9, 0.999)),
            beta3=config.get('beta3', None),
            weight_decay=config.get('weight_decay', 0),
            d0=config.get('d0', 1e-6),
        )

    elif optimizer_name == 'Adan':
        try:
            from adan import Adan
        except ImportError:
            print("Installing adan-optimizer...")
            os.system("pip install git+https://github.com/sail-sg/Adan.git")
            from adan import Adan

        return Adan(
            model.parameters(),
            lr=config['lr'],
            betas=config.get('betas', (0.98, 0.92, 0.99)),
            weight_decay=config.get('weight_decay', 0),
        )

    elif optimizer_name == 'LAMB':
        try:
            from pytorch_lamb import Lamb
        except ImportError:
            print("Installing pytorch-lamb...")
            os.system("pip install pytorch-lamb")
            from pytorch_lamb import Lamb

        return Lamb(
            model.parameters(),
            lr=config['lr'],
            betas=config.get('betas', (0.9, 0.999)),
            weight_decay=config.get('weight_decay', 0),
        )

    elif optimizer_name == 'Novograd':
        try:
            from nvidia_novograd import Novograd
        except ImportError:
            print("Installing nvidia-novograd...")
            os.system("pip install nvidia-novograd")
            from nvidia_novograd import Novograd

        return Novograd(
            model.parameters(),
            lr=config['lr'],
            betas=config.get('betas', (0.95, 0.98)),
            weight_decay=config.get('weight_decay', 0),
        )

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_scheduler(optimizer, scheduler_config, nBatch):
    """
    Returns the learning rate scheduler based on configuration.
    """
    scheduler_type = scheduler_config['type']

    if scheduler_type == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.get('T_max') or nBatch,
            eta_min=scheduler_config.get('eta_min', 0),
        )

    elif scheduler_type == 'CosineAnnealingWarmRestarts':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_config.get('T_0', 1000),
            T_mult=scheduler_config.get('T_mult', 2),
            eta_min=scheduler_config.get('eta_min', 0),
        )

    elif scheduler_type == 'OneCycleLR':
        max_lr = scheduler_config.get('max_lr') or optimizer.param_groups[0]['lr']
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=scheduler_config.get('total_steps') or nBatch,
            pct_start=scheduler_config.get('pct_start', 0.3),
            anneal_strategy=scheduler_config.get('anneal_strategy', 'cos'),
        )

    elif scheduler_type == 'PolynomialLR':
        return torch.optim.lr_scheduler.PolynomialLR(
            optimizer,
            total_iters=scheduler_config.get('total_iters') or nBatch,
            power=scheduler_config.get('power', 1.0),
        )

    elif scheduler_type == 'ExponentialLR':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=scheduler_config.get('gamma', 0.95),
        )

    elif scheduler_type == 'LinearLR':
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=scheduler_config.get('start_factor', 1.0),
            end_factor=scheduler_config.get('end_factor', 0.1),
            total_iters=scheduler_config.get('total_iters') or nBatch,
        )

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def wrap_with_lookahead(optimizer, config):
    """
    Wraps optimizer with Lookahead if specified in config.
    """
    if not config.get('use_lookahead', False):
        return optimizer

    try:
        from pytorch_optimizer import Lookahead
    except ImportError:
        print("Installing pytorch-optimizer for Lookahead...")
        os.system("pip install pytorch-optimizer")
        from pytorch_optimizer import Lookahead

    return Lookahead(
        optimizer,
        k=config.get('lookahead_k', 5),
        alpha=config.get('lookahead_alpha', 0.5),
    )


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_single_config(
    base_args: Dict[str, Any],
    opt_config: Dict[str, Any],
    scheduler_config: Dict[str, Any],
    run_id: int,
) -> Dict[str, Any]:
    """
    Train a single model with given optimizer and scheduler configuration.

    Returns:
        Dictionary containing training results and metadata.
    """
    import time
    import pickle
    from torch.nn.utils.rnn import pad_sequence
    from edit_distance import SequenceMatcher

    # Setup
    run_name = f"{opt_config['name']}_{scheduler_config['name']}_run{run_id}"
    output_dir = os.path.join(base_args['sweep_output_dir'], run_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Starting: {run_name}")
    print(f"{'='*80}\n")

    device = base_args.get('device', 'cuda')
    torch.manual_seed(base_args['seed'])
    np.random.seed(base_args['seed'])

    # Save configuration
    config_dict = {
        'base_args': base_args,
        'optimizer_config': opt_config,
        'scheduler_config': scheduler_config,
        'run_id': run_id,
        'timestamp': datetime.datetime.now().isoformat(),
    }
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)

    # Load data
    trainLoader, testLoader, loadedData = getDatasetLoaders(
        base_args['datasetPath'],
        base_args['batchSize'],
    )

    # Create model
    model = GRUDecoder(
        neural_dim=base_args['nInputFeatures'],
        n_classes=base_args['nClasses'],
        hidden_dim=base_args['nUnits'],
        layer_dim=base_args['nLayers'],
        nDays=len(loadedData['train']),
        dropout=base_args['dropout'],
        device=device,
        strideLen=base_args['strideLen'],
        kernelLen=base_args['kernelLen'],
        gaussianSmoothWidth=base_args['gaussianSmoothWidth'],
        bidirectional=base_args['bidirectional'],
    ).to(device)

    # Create optimizer
    optimizer = get_optimizer(model, opt_config)
    optimizer = wrap_with_lookahead(optimizer, opt_config)

    # Create scheduler
    scheduler = get_scheduler(optimizer, scheduler_config, base_args['nBatch'])

    # Setup SWA if requested
    swa_model = None
    swa_scheduler = None
    if opt_config.get('use_swa', False):
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        swa_start = int(base_args['nBatch'] * opt_config.get('swa_start', 0.75))
        swa_scheduler = torch.optim.swa_utils.SWALR(
            optimizer,
            swa_lr=opt_config['lr'] * 0.1,
        )

    # Loss function
    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    # Training loop
    testLoss = []
    testCER = []
    trainLoss = []
    learning_rates = []
    startTime = time.time()
    best_cer = float('inf')

    for batch in range(base_args['nBatch']):
        model.train()

        # Get batch
        X, y, X_len, y_len, dayIdx = next(iter(trainLoader))
        X, y, X_len, y_len, dayIdx = (
            X.to(device),
            y.to(device),
            X_len.to(device),
            y_len.to(device),
            dayIdx.to(device),
        )

        # Data augmentation
        if base_args['whiteNoiseSD'] > 0:
            X += torch.randn(X.shape, device=device) * base_args['whiteNoiseSD']

        if base_args['constantOffsetSD'] > 0:
            X += (
                torch.randn([X.shape[0], 1, X.shape[2]], device=device)
                * base_args['constantOffsetSD']
            )

        # Forward pass
        pred = model.forward(X, dayIdx)

        loss = loss_ctc(
            torch.permute(pred.log_softmax(2), [1, 0, 2]),
            y,
            ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
            y_len,
        )
        loss = torch.sum(loss)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if base_args.get('grad_clip', None):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                base_args['grad_clip']
            )

        optimizer.step()

        # Update learning rate
        if swa_model and batch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        trainLoss.append(loss.item())
        learning_rates.append(optimizer.param_groups[0]['lr'])

        # Evaluation
        if batch % base_args.get('eval_every', 100) == 0:
            eval_model = swa_model if (swa_model and batch >= swa_start) else model

            with torch.no_grad():
                eval_model.eval()
                allLoss = []
                total_edit_distance = 0
                total_seq_length = 0

                for X, y, X_len, y_len, testDayIdx in testLoader:
                    X, y, X_len, y_len, testDayIdx = (
                        X.to(device),
                        y.to(device),
                        X_len.to(device),
                        y_len.to(device),
                        testDayIdx.to(device),
                    )

                    pred = eval_model.forward(X, testDayIdx)
                    loss = loss_ctc(
                        torch.permute(pred.log_softmax(2), [1, 0, 2]),
                        y,
                        ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
                        y_len,
                    )
                    loss = torch.sum(loss)
                    allLoss.append(loss.cpu().detach().numpy())

                    adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(
                        torch.int32
                    )
                    for iterIdx in range(pred.shape[0]):
                        decodedSeq = torch.argmax(
                            torch.tensor(pred[iterIdx, 0 : adjustedLens[iterIdx], :]),
                            dim=-1,
                        )
                        decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                        decodedSeq = decodedSeq.cpu().detach().numpy()
                        decodedSeq = np.array([i for i in decodedSeq if i != 0])

                        trueSeq = np.array(
                            y[iterIdx][0 : y_len[iterIdx]].cpu().detach()
                        )

                        matcher = SequenceMatcher(
                            a=trueSeq.tolist(), b=decodedSeq.tolist()
                        )
                        total_edit_distance += matcher.distance()
                        total_seq_length += len(trueSeq)

                avgTestLoss = np.sum(allLoss) / len(testLoader)
                cer = total_edit_distance / total_seq_length

                endTime = time.time()
                elapsed = endTime - startTime

                print(
                    f"[{run_name}] Batch {batch}/{base_args['nBatch']} | "
                    f"Train Loss: {trainLoss[-1]:.4f} | "
                    f"Test Loss: {avgTestLoss:.4f} | "
                    f"CER: {cer:.4f} | "
                    f"LR: {learning_rates[-1]:.6f} | "
                    f"Time: {elapsed:.2f}s"
                )

                testLoss.append(avgTestLoss)
                testCER.append(cer)

                # Save best model
                if cer < best_cer:
                    best_cer = cer
                    save_model = swa_model if swa_model else model
                    torch.save(
                        save_model.state_dict(),
                        os.path.join(output_dir, 'best_model.pt')
                    )

                startTime = time.time()

    # Finalize SWA if used
    if swa_model:
        torch.optim.swa_utils.update_bn(trainLoader, swa_model, device=device)

    # Calculate comprehensive metrics
    total_time = time.time() - startTime

    # Compute additional metrics for comparison
    avg_train_loss = np.mean(trainLoss)
    final_train_loss = trainLoss[-1] if trainLoss else float('inf')
    avg_test_loss = np.mean(testLoss) if testLoss else float('inf')
    final_test_loss = testLoss[-1] if testLoss else float('inf')
    avg_cer = np.mean(testCER) if testCER else float('inf')
    final_cer = testCER[-1] if testCER else float('inf')
    min_cer = best_cer

    # Calculate improvement metrics
    if len(testCER) > 1:
        cer_improvement = testCER[0] - testCER[-1]
        cer_improvement_pct = (cer_improvement / testCER[0] * 100) if testCER[0] > 0 else 0
    else:
        cer_improvement = 0
        cer_improvement_pct = 0

    # Save final results with comprehensive metrics
    results = {
        'testLoss': np.array(testLoss),
        'testCER': np.array(testCER),
        'trainLoss': np.array(trainLoss),
        'learning_rates': np.array(learning_rates),

        # Key comparison metrics
        'best_cer': best_cer,
        'final_cer': final_cer,
        'avg_cer': avg_cer,
        'avg_train_loss': avg_train_loss,
        'final_train_loss': final_train_loss,
        'avg_test_loss': avg_test_loss,
        'final_test_loss': final_test_loss,
        'cer_improvement': cer_improvement,
        'cer_improvement_pct': cer_improvement_pct,
        'total_time_seconds': total_time,

        # Configuration
        'optimizer_config': opt_config,
        'scheduler_config': scheduler_config,
        'run_name': run_name,
    }

    with open(os.path.join(output_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # Save final model
    final_model = swa_model if swa_model else model
    torch.save(
        final_model.state_dict(),
        os.path.join(output_dir, 'final_model.pt')
    )

    print(f"\n{'='*80}")
    print(f"Completed: {run_name}")
    print(f"Best CER: {best_cer:.6f}")
    print(f"Final CER: {final_cer:.6f}")
    print(f"Average CER: {avg_cer:.6f}")
    print(f"CER Improvement: {cer_improvement:.6f} ({cer_improvement_pct:.2f}%)")
    print(f"Total Time: {total_time:.2f}s")
    print(f"{'='*80}\n")

    # Clean up memory
    del model
    del optimizer
    del scheduler
    if swa_model:
        del swa_model
    if swa_scheduler:
        del swa_scheduler
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    return results


# ============================================================================
# MAIN SWEEP EXECUTION
# ============================================================================

def run_optimizer_sweep(
    base_args: Dict[str, Any],
    optimizer_configs: List[Dict[str, Any]] = None,
    scheduler_configs: List[Dict[str, Any]] = None,
):
    """
    Run the full optimizer sweep.

    Args:
        base_args: Base training arguments
        optimizer_configs: List of optimizer configurations (if None, uses all)
        scheduler_configs: List of scheduler configurations (if None, uses all)
    """
    if optimizer_configs is None:
        optimizer_configs = get_optimizer_configs()

    if scheduler_configs is None:
        scheduler_configs = get_scheduler_configs()

    # Create all combinations
    all_configs = list(itertools.product(optimizer_configs, scheduler_configs))

    print(f"\n{'='*80}")
    print(f"OPTIMIZER SWEEP")
    print(f"{'='*80}")
    print(f"Total configurations: {len(all_configs)}")
    print(f"Optimizer configs: {len(optimizer_configs)}")
    print(f"Scheduler configs: {len(scheduler_configs)}")
    print(f"Output directory: {base_args['sweep_output_dir']}")
    print(f"{'='*80}\n")

    # Run all configurations
    all_results = []
    start_run_id = 27  # Resume from run 26
    for run_id, (opt_config, sched_config) in enumerate(all_configs):
        # Skip runs before start_run_id
        if run_id < start_run_id:
            print(f"Skipping run {run_id} (already completed)")
            continue

        try:
            results = train_single_config(
                base_args=base_args,
                opt_config=opt_config,
                scheduler_config=sched_config,
                run_id=run_id,
            )
            all_results.append(results)
        except Exception as e:
            print(f"ERROR in run {run_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Create comprehensive comparison metrics
    comparison_metrics = []
    for result in all_results:
        comparison_metrics.append({
            'run_name': result['run_name'],
            'optimizer': result['optimizer_config']['name'],
            'scheduler': result['scheduler_config']['name'],
            'best_cer': result['best_cer'],
            'final_cer': result['final_cer'],
            'avg_cer': result['avg_cer'],
            'avg_train_loss': result['avg_train_loss'],
            'final_train_loss': result['final_train_loss'],
            'avg_test_loss': result['avg_test_loss'],
            'final_test_loss': result['final_test_loss'],
            'cer_improvement': result['cer_improvement'],
            'cer_improvement_pct': result['cer_improvement_pct'],
            'total_time_seconds': result['total_time_seconds'],
        })

    # Sort by best CER
    comparison_metrics_sorted = sorted(comparison_metrics, key=lambda x: x['best_cer'])

    # Save summary
    summary = {
        'total_runs': len(all_results),
        'best_run': min(all_results, key=lambda x: x['best_cer']) if all_results else None,
        'all_results': all_results,
        'comparison_metrics': comparison_metrics_sorted,
    }

    with open(os.path.join(base_args['sweep_output_dir'], 'summary.pkl'), 'wb') as f:
        pickle.dump(summary, f)

    # Save comparison metrics as CSV for easy viewing
    import csv
    csv_path = os.path.join(base_args['sweep_output_dir'], 'comparison_metrics.csv')
    if comparison_metrics_sorted:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=comparison_metrics_sorted[0].keys())
            writer.writeheader()
            writer.writerows(comparison_metrics_sorted)
        print(f"Saved comparison metrics to: {csv_path}")

    # Save comparison metrics as JSON for readability
    json_path = os.path.join(base_args['sweep_output_dir'], 'comparison_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(comparison_metrics_sorted, f, indent=2)
    print(f"Saved comparison metrics to: {json_path}")

    # Print summary
    print(f"\n{'='*80}")
    print("SWEEP COMPLETE")
    print(f"{'='*80}")
    if all_results:
        best = summary['best_run']
        print(f"Best configuration:")
        print(f"  Optimizer: {best['optimizer_config']['name']}")
        print(f"  Scheduler: {best['scheduler_config']['name']}")
        print(f"  Best CER: {best['best_cer']:.6f}")
        print(f"  Final CER: {best['final_cer']:.6f}")
        print(f"  Average CER: {best['avg_cer']:.6f}")
        print(f"  CER Improvement: {best['cer_improvement']:.6f} ({best['cer_improvement_pct']:.2f}%)")
        print(f"  Total Time: {best['total_time_seconds']:.2f}s")
        print(f"\nTop 5 configurations by Best CER:")
        for i, metrics in enumerate(comparison_metrics_sorted[:5], 1):
            print(f"  {i}. {metrics['optimizer']} + {metrics['scheduler']}")
            print(f"     Best CER: {metrics['best_cer']:.6f} | Final: {metrics['final_cer']:.6f} | Avg: {metrics['avg_cer']:.6f}")
    print(f"{'='*80}\n")


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    # Base configuration
    base_args = {
        'sweep_output_dir': '/home/iseanbhanot/optimizer_sweep_results',
        'datasetPath': '/home/iseanbhanot/dataset/ptDecoder_ctc.pkl',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',

        # Model architecture
        'nInputFeatures': 256,
        'nClasses': 40,
        'nUnits': 1024,
        'nLayers': 5,
        'dropout': 0.4,
        'bidirectional': True,

        # Data processing
        'seqLen': 150,
        'maxTimeSeriesLen': 1200,
        'strideLen': 4,
        'kernelLen': 32,
        'gaussianSmoothWidth': 2.0,

        # Data augmentation
        'whiteNoiseSD': 0.8,
        'constantOffsetSD': 0.2,

        # Training
        'batchSize': 32,
        'nBatch': 2500,  # Reduced from 5000 to half
        'seed': 0,
        'eval_every': 100,
        'grad_clip': 1.0,  # Gradient clipping
    }

    # Get configurations
    optimizer_configs = get_optimizer_configs()
    scheduler_configs = get_scheduler_configs()

    # Run full sweep with all configurations
    run_optimizer_sweep(
        base_args=base_args,
        optimizer_configs=optimizer_configs,
        scheduler_configs=scheduler_configs,
    )