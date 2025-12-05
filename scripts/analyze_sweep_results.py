"""
Analysis and visualization script for optimizer sweep results.

This script:
- Loads all results from the sweep
- Generates comparison plots
- Creates ranking tables
- Identifies best configurations
"""

import os
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any


def load_sweep_results(sweep_dir: str) -> Dict[str, Any]:
    """
    Load all results from a sweep directory.

    Args:
        sweep_dir: Path to the sweep output directory

    Returns:
        Dictionary containing all results and metadata
    """
    results = []

    # Iterate through all subdirectories
    for run_dir in sorted(Path(sweep_dir).iterdir()):
        if not run_dir.is_dir():
            continue

        results_file = run_dir / 'results.pkl'
        config_file = run_dir / 'config.json'

        if not results_file.exists() or not config_file.exists():
            print(f"Skipping incomplete run: {run_dir.name}")
            continue

        # Load results
        with open(results_file, 'rb') as f:
            run_results = pickle.load(f)

        # Load config
        with open(config_file, 'r') as f:
            config = json.load(f)

        # Combine
        run_data = {
            'run_name': run_dir.name,
            'run_dir': str(run_dir),
            'config': config,
            'results': run_results,
        }
        results.append(run_data)

    return {
        'sweep_dir': sweep_dir,
        'num_runs': len(results),
        'runs': results,
    }


def create_ranking_table(sweep_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a ranking table of all configurations.

    Args:
        sweep_data: Dictionary from load_sweep_results

    Returns:
        Pandas DataFrame with rankings
    """
    rows = []

    for run in sweep_data['runs']:
        opt_config = run['config']['optimizer_config']
        sched_config = run['config']['scheduler_config']
        results = run['results']

        row = {
            'run_name': run['run_name'],
            'optimizer': opt_config.get('optimizer', 'Unknown'),
            'scheduler': sched_config.get('name', 'Unknown'),
            'lr': opt_config.get('lr', None),
            'weight_decay': opt_config.get('weight_decay', None),
            'best_cer': results.get('best_cer', float('inf')),
            'final_cer': results.get('final_cer', float('inf')),
            'best_test_loss': np.min(results.get('testLoss', [float('inf')])),
            'final_test_loss': results.get('testLoss', [float('inf')])[-1],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values('best_cer')
    df['rank'] = range(1, len(df) + 1)

    return df


def plot_training_curves(sweep_data: Dict[str, Any], top_k: int = 10, output_dir: str = None):
    """
    Plot training curves for top K configurations.

    Args:
        sweep_data: Dictionary from load_sweep_results
        top_k: Number of top configurations to plot
        output_dir: Directory to save plots (if None, just display)
    """
    # Sort runs by best CER
    sorted_runs = sorted(
        sweep_data['runs'],
        key=lambda x: x['results'].get('best_cer', float('inf'))
    )[:top_k]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Top {top_k} Optimizer Configurations', fontsize=16, fontweight='bold')

    # Plot 1: Test CER over time
    ax1 = axes[0, 0]
    for run in sorted_runs:
        results = run['results']
        cer = results.get('testCER', [])
        if len(cer) > 0:
            ax1.plot(cer, label=run['run_name'], linewidth=2, alpha=0.7)
    ax1.set_xlabel('Evaluation Step', fontsize=12)
    ax1.set_ylabel('Character Error Rate (CER)', fontsize=12)
    ax1.set_title('Test CER vs Training Progress', fontsize=14)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Test Loss over time
    ax2 = axes[0, 1]
    for run in sorted_runs:
        results = run['results']
        loss = results.get('testLoss', [])
        if len(loss) > 0:
            ax2.plot(loss, label=run['run_name'], linewidth=2, alpha=0.7)
    ax2.set_xlabel('Evaluation Step', fontsize=12)
    ax2.set_ylabel('Test Loss', fontsize=12)
    ax2.set_title('Test Loss vs Training Progress', fontsize=14)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Learning Rate Schedule
    ax3 = axes[1, 0]
    for run in sorted_runs:
        results = run['results']
        lr = results.get('learning_rates', [])
        if len(lr) > 0:
            ax3.plot(lr, label=run['run_name'], linewidth=2, alpha=0.7)
    ax3.set_xlabel('Training Step', fontsize=12)
    ax3.set_ylabel('Learning Rate', fontsize=12)
    ax3.set_title('Learning Rate Schedule', fontsize=14)
    ax3.set_yscale('log')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Final Performance Comparison
    ax4 = axes[1, 1]
    run_names = [run['run_name'][:30] for run in sorted_runs]  # Truncate names
    best_cers = [run['results'].get('best_cer', float('inf')) for run in sorted_runs]
    final_cers = [run['results'].get('final_cer', float('inf')) for run in sorted_runs]

    x = np.arange(len(run_names))
    width = 0.35

    ax4.bar(x - width/2, best_cers, width, label='Best CER', alpha=0.8)
    ax4.bar(x + width/2, final_cers, width, label='Final CER', alpha=0.8)
    ax4.set_xlabel('Configuration', fontsize=12)
    ax4.set_ylabel('Character Error Rate', fontsize=12)
    ax4.set_title('Best vs Final CER Comparison', fontsize=14)
    ax4.set_xticks(x)
    ax4.set_xticklabels(run_names, rotation=45, ha='right', fontsize=8)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        print(f"Saved training curves to {output_dir}/training_curves.png")
    else:
        plt.show()

    plt.close()


def plot_optimizer_comparison(sweep_data: Dict[str, Any], output_dir: str = None):
    """
    Create box plots comparing different optimizer families.

    Args:
        sweep_data: Dictionary from load_sweep_results
        output_dir: Directory to save plots
    """
    # Group results by optimizer type
    optimizer_results = {}

    for run in sweep_data['runs']:
        opt_name = run['config']['optimizer_config'].get('optimizer', 'Unknown')
        best_cer = run['results'].get('best_cer', float('inf'))

        if opt_name not in optimizer_results:
            optimizer_results[opt_name] = []
        optimizer_results[opt_name].append(best_cer)

    # Create box plot
    fig, ax = plt.subplots(figsize=(12, 6))

    optimizers = list(optimizer_results.keys())
    data = [optimizer_results[opt] for opt in optimizers]

    bp = ax.boxplot(data, labels=optimizers, patch_artist=True, showmeans=True)

    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(optimizers)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_xlabel('Optimizer', fontsize=12)
    ax.set_ylabel('Best CER', fontsize=12)
    ax.set_title('Optimizer Performance Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'optimizer_comparison.png'), dpi=300, bbox_inches='tight')
        print(f"Saved optimizer comparison to {output_dir}/optimizer_comparison.png")
    else:
        plt.show()

    plt.close()


def plot_scheduler_comparison(sweep_data: Dict[str, Any], output_dir: str = None):
    """
    Create box plots comparing different scheduler types.

    Args:
        sweep_data: Dictionary from load_sweep_results
        output_dir: Directory to save plots
    """
    # Group results by scheduler type
    scheduler_results = {}

    for run in sweep_data['runs']:
        sched_name = run['config']['scheduler_config'].get('name', 'Unknown')
        best_cer = run['results'].get('best_cer', float('inf'))

        if sched_name not in scheduler_results:
            scheduler_results[sched_name] = []
        scheduler_results[sched_name].append(best_cer)

    # Create box plot
    fig, ax = plt.subplots(figsize=(12, 6))

    schedulers = list(scheduler_results.keys())
    data = [scheduler_results[sched] for sched in schedulers]

    bp = ax.boxplot(data, labels=schedulers, patch_artist=True, showmeans=True)

    # Color the boxes
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(schedulers)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_xlabel('Scheduler', fontsize=12)
    ax.set_ylabel('Best CER', fontsize=12)
    ax.set_title('Scheduler Performance Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'scheduler_comparison.png'), dpi=300, bbox_inches='tight')
        print(f"Saved scheduler comparison to {output_dir}/scheduler_comparison.png")
    else:
        plt.show()

    plt.close()


def plot_hyperparameter_heatmap(sweep_data: Dict[str, Any], output_dir: str = None):
    """
    Create heatmaps showing hyperparameter effects.

    Args:
        sweep_data: Dictionary from load_sweep_results
        output_dir: Directory to save plots
    """
    # Extract learning rate and weight decay for each run
    lr_wd_results = {}

    for run in sweep_data['runs']:
        opt_config = run['config']['optimizer_config']
        lr = opt_config.get('lr', None)
        wd = opt_config.get('weight_decay', None)
        best_cer = run['results'].get('best_cer', float('inf'))

        if lr is not None and wd is not None:
            key = (lr, wd)
            if key not in lr_wd_results:
                lr_wd_results[key] = []
            lr_wd_results[key].append(best_cer)

    if not lr_wd_results:
        print("No runs with both LR and WD found for heatmap")
        return

    # Average results for each (lr, wd) pair
    lr_wd_avg = {k: np.mean(v) for k, v in lr_wd_results.items()}

    # Get unique values
    lrs = sorted(set(k[0] for k in lr_wd_avg.keys()))
    wds = sorted(set(k[1] for k in lr_wd_avg.keys()))

    # Create matrix
    matrix = np.full((len(wds), len(lrs)), np.nan)
    for (lr, wd), cer in lr_wd_avg.items():
        i = wds.index(wd)
        j = lrs.index(lr)
        matrix[i, j] = cer

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto')

    ax.set_xticks(np.arange(len(lrs)))
    ax.set_yticks(np.arange(len(wds)))
    ax.set_xticklabels([f'{lr:.1e}' for lr in lrs])
    ax.set_yticklabels([f'{wd:.1e}' for wd in wds])

    ax.set_xlabel('Learning Rate', fontsize=12)
    ax.set_ylabel('Weight Decay', fontsize=12)
    ax.set_title('Best CER Heatmap (Learning Rate vs Weight Decay)', fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Best CER', fontsize=12)

    # Add text annotations
    for i in range(len(wds)):
        for j in range(len(lrs)):
            if not np.isnan(matrix[i, j]):
                text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'hyperparameter_heatmap.png'), dpi=300, bbox_inches='tight')
        print(f"Saved hyperparameter heatmap to {output_dir}/hyperparameter_heatmap.png")
    else:
        plt.show()

    plt.close()


def generate_report(sweep_dir: str, output_dir: str = None, top_k: int = 10):
    """
    Generate a comprehensive analysis report.

    Args:
        sweep_dir: Path to sweep results directory
        output_dir: Directory to save analysis outputs
        top_k: Number of top configurations to highlight
    """
    if output_dir is None:
        output_dir = os.path.join(sweep_dir, 'analysis')

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print("LOADING SWEEP RESULTS")
    print(f"{'='*80}\n")

    # Load results
    sweep_data = load_sweep_results(sweep_dir)
    print(f"Loaded {sweep_data['num_runs']} runs")

    if sweep_data['num_runs'] == 0:
        print("No valid runs found!")
        return

    # Create ranking table
    print(f"\n{'='*80}")
    print("GENERATING RANKING TABLE")
    print(f"{'='*80}\n")

    ranking_df = create_ranking_table(sweep_data)

    # Save ranking table
    ranking_df.to_csv(os.path.join(output_dir, 'rankings.csv'), index=False)
    print(f"Saved rankings to {output_dir}/rankings.csv")

    # Print top K
    print(f"\nTop {top_k} Configurations:")
    print(ranking_df.head(top_k).to_string(index=False))

    # Generate plots
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*80}\n")

    plot_training_curves(sweep_data, top_k=top_k, output_dir=output_dir)
    plot_optimizer_comparison(sweep_data, output_dir=output_dir)
    plot_scheduler_comparison(sweep_data, output_dir=output_dir)
    plot_hyperparameter_heatmap(sweep_data, output_dir=output_dir)

    # Generate summary report
    print(f"\n{'='*80}")
    print("GENERATING SUMMARY REPORT")
    print(f"{'='*80}\n")

    report_lines = [
        "=" * 80,
        "OPTIMIZER SWEEP ANALYSIS REPORT",
        "=" * 80,
        f"\nGenerated: {pd.Timestamp.now()}",
        f"Sweep Directory: {sweep_dir}",
        f"Total Runs: {sweep_data['num_runs']}",
        "\n" + "=" * 80,
        "TOP 10 CONFIGURATIONS",
        "=" * 80,
        "\n" + ranking_df.head(10).to_string(index=False),
        "\n" + "=" * 80,
        "OPTIMIZER SUMMARY",
        "=" * 80,
    ]

    # Optimizer statistics
    opt_stats = ranking_df.groupby('optimizer')['best_cer'].agg(['mean', 'min', 'count'])
    opt_stats = opt_stats.sort_values('mean')
    report_lines.append("\n" + opt_stats.to_string())

    report_lines.extend([
        "\n" + "=" * 80,
        "SCHEDULER SUMMARY",
        "=" * 80,
    ])

    # Scheduler statistics
    sched_stats = ranking_df.groupby('scheduler')['best_cer'].agg(['mean', 'min', 'count'])
    sched_stats = sched_stats.sort_values('mean')
    report_lines.append("\n" + sched_stats.to_string())

    report_lines.append("\n" + "=" * 80)

    # Save report
    report_text = '\n'.join(report_lines)
    with open(os.path.join(output_dir, 'report.txt'), 'w') as f:
        f.write(report_text)

    print(report_text)
    print(f"\nSaved report to {output_dir}/report.txt")
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Analyze optimizer sweep results')
    parser.add_argument(
        '--sweep-dir',
        type=str,
        default='/home/iseanbhanot/optimizer_sweep_results',
        help='Directory containing sweep results'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save analysis outputs (default: sweep_dir/analysis)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of top configurations to highlight'
    )

    args = parser.parse_args()

    generate_report(
        sweep_dir=args.sweep_dir,
        output_dir=args.output_dir,
        top_k=args.top_k,
    )
