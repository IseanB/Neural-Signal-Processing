"""
Enhanced visualization script for optimizer sweep results.

This script creates comprehensive, publication-quality visualizations including:
- Interactive performance dashboards
- Optimizer family comparisons
- Convergence analysis
- Hyperparameter sensitivity analysis
- Statistical significance tests
- Category-based performance breakdowns
"""

import os
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
from scipy import stats
from matplotlib.gridspec import GridSpec

# Set style for publication-quality plots
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'


def load_all_results(sweep_dir: str) -> pd.DataFrame:
    rows = []

    for run_dir in sorted(Path(sweep_dir).iterdir()):
        if not run_dir.is_dir():
            continue

        results_file = run_dir / 'results.pkl'
        config_file = run_dir / 'config.json'

        if not results_file.exists() or not config_file.exists():
            print(f"Skipping incomplete run: {run_dir.name}")
            continue

        # Load data
        with open(results_file, 'rb') as f:
            results = pickle.load(f)

        with open(config_file, 'r') as f:
            config = json.load(f)

        # Extract key information
        opt_config = config['optimizer_config']
        sched_config = config['scheduler_config']

        row = {
            'run_name': run_dir.name,
            'run_dir': str(run_dir),
            'run_id': config.get('run_id', -1),

            # Optimizer info
            'optimizer': opt_config.get('optimizer', 'Unknown'),
            'optimizer_name': opt_config.get('name', 'Unknown'),
            'lr': opt_config.get('lr', None),
            'weight_decay': opt_config.get('weight_decay', None),
            'betas': str(opt_config.get('betas', None)),

            # Scheduler info
            'scheduler': sched_config.get('name', 'Unknown'),
            'scheduler_type': sched_config.get('type', 'Unknown'),

            # Performance metrics
            'best_cer': results.get('best_cer', float('inf')),
            'final_cer': results.get('final_cer', float('inf')),
            'avg_cer': results.get('avg_cer', float('inf')),
            'cer_improvement': results.get('cer_improvement', 0),
            'cer_improvement_pct': results.get('cer_improvement_pct', 0),

            # Loss metrics
            'avg_train_loss': results.get('avg_train_loss', float('inf')),
            'final_train_loss': results.get('final_train_loss', float('inf')),
            'avg_test_loss': results.get('avg_test_loss', float('inf')),
            'final_test_loss': results.get('final_test_loss', float('inf')),

            # Training info
            'total_time_seconds': results.get('total_time_seconds', 0),

            # Full arrays for detailed analysis
            'testCER': results.get('testCER', np.array([])),
            'testLoss': results.get('testLoss', np.array([])),
            'trainLoss': results.get('trainLoss', np.array([])),
            'learning_rates': results.get('learning_rates', np.array([])),
        }

        # Categorize optimizer
        row['optimizer_category'] = categorize_optimizer(opt_config['optimizer'])
        row['is_adaptive'] = is_adaptive_optimizer(opt_config['optimizer'])

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by best CER
    df = df.sort_values('best_cer').reset_index(drop=True)
    df['rank'] = df.index + 1

    return df


def categorize_optimizer(optimizer_name: str) -> str:
    """Categorize optimizer into families."""
    categories = {
        'Adam-family': ['AdamW', 'Adam', 'AdEMAMix', 'Adan'],
        'Second-order': ['Sophia', 'Shampoo'],
        'Novel': ['Lion', 'Prodigy'],
        'Large-batch': ['LAMB', 'LARS'],
        'Classic': ['SGD', 'Novograd'],
    }

    for category, opts in categories.items():
        if optimizer_name in opts:
            return category
    return 'Other'


def is_adaptive_optimizer(optimizer_name: str) -> bool:
    """Check if optimizer is adaptive (auto-tunes LR)."""
    adaptive = ['Prodigy', 'Sophia', 'Lion']
    return optimizer_name in adaptive


def create_comprehensive_dashboard(df: pd.DataFrame, output_dir: str):
    """
    Create a comprehensive dashboard with multiple visualizations.

    Args:
        df: DataFrame with all results
        output_dir: Directory to save plots
    """
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Top 10 configurations comparison
    ax1 = fig.add_subplot(gs[0, :2])
    top10 = df.head(10)
    colors = plt.cm.viridis(np.linspace(0, 1, len(top10)))
    bars = ax1.barh(range(len(top10)), top10['best_cer'], color=colors)
    ax1.set_yticks(range(len(top10)))
    ax1.set_yticklabels([f"{row['optimizer']} + {row['scheduler']}"
                          for _, row in top10.iterrows()], fontsize=9)
    ax1.set_xlabel('Best CER', fontweight='bold')
    ax1.set_title('Top 10 Configurations by Best CER', fontweight='bold', fontsize=14)
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)

    # Add values on bars
    for i, (bar, val) in enumerate(zip(bars, top10['best_cer'])):
        ax1.text(val, i, f' {val:.4f}', va='center', fontsize=8)

    # 2. Optimizer category comparison (boxplot)
    ax2 = fig.add_subplot(gs[0, 2])
    df_valid = df[df['best_cer'] != float('inf')]
    if len(df_valid) > 0:
        category_data = [df_valid[df_valid['optimizer_category'] == cat]['best_cer'].values
                        for cat in df_valid['optimizer_category'].unique()]
        bp = ax2.boxplot(category_data,
                        tick_labels=df_valid['optimizer_category'].unique(),
                        patch_artist=True, showmeans=True, meanline=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        ax2.set_ylabel('Best CER', fontweight='bold')
        ax2.set_title('Performance by Optimizer Category', fontweight='bold', fontsize=12)
        ax2.tick_params(axis='x', rotation=45, labelsize=8)
        ax2.grid(axis='y', alpha=0.3)

    # 3. Scheduler comparison (violin plot)
    ax3 = fig.add_subplot(gs[1, 0])
    if len(df_valid) > 0:
        schedulers = df_valid['scheduler'].unique()
        positions = range(len(schedulers))
        violin_data = [df_valid[df_valid['scheduler'] == s]['best_cer'].values
                      for s in schedulers]
        parts = ax3.violinplot(violin_data, positions=positions, showmeans=True, showmedians=True)
        ax3.set_xticks(positions)
        ax3.set_xticklabels(schedulers, rotation=45, ha='right', fontsize=9)
        ax3.set_ylabel('Best CER', fontweight='bold')
        ax3.set_title('Scheduler Performance Distribution', fontweight='bold', fontsize=12)
        ax3.grid(axis='y', alpha=0.3)

    # 4. Learning rate vs performance scatter
    ax4 = fig.add_subplot(gs[1, 1])
    df_lr = df_valid[df_valid['lr'].notna() & (df_valid['lr'] < 1.0)]  # Exclude Prodigy
    if len(df_lr) > 0:
        scatter = ax4.scatter(df_lr['lr'], df_lr['best_cer'],
                             c=df_lr['rank'], cmap='RdYlGn_r',
                             s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax4.set_xlabel('Learning Rate', fontweight='bold')
        ax4.set_ylabel('Best CER', fontweight='bold')
        ax4.set_xscale('log')
        ax4.set_title('Learning Rate vs Performance', fontweight='bold', fontsize=12)
        ax4.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax4, label='Rank')

    # 5. Training efficiency (time vs performance)
    ax5 = fig.add_subplot(gs[1, 2])
    if len(df_valid) > 0:
        scatter = ax5.scatter(df_valid['total_time_seconds'], df_valid['best_cer'],
                             c=df_valid['rank'], cmap='RdYlGn_r',
                             s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax5.set_xlabel('Training Time (seconds)', fontweight='bold')
        ax5.set_ylabel('Best CER', fontweight='bold')
        ax5.set_title('Training Efficiency', fontweight='bold', fontsize=12)
        ax5.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax5, label='Rank')

    # 6. Convergence curves for top 5
    ax6 = fig.add_subplot(gs[2, :])
    top5 = df.head(5)
    for idx, row in top5.iterrows():
        if len(row['testCER']) > 0:
            ax6.plot(row['testCER'], label=f"{row['optimizer']} + {row['scheduler']}",
                    linewidth=2, alpha=0.8)
    ax6.set_xlabel('Evaluation Step', fontweight='bold')
    ax6.set_ylabel('Test CER', fontweight='bold')
    ax6.set_title('Convergence Curves - Top 5 Configurations', fontweight='bold', fontsize=14)
    ax6.legend(loc='best', fontsize=9)
    ax6.grid(alpha=0.3)

    plt.suptitle('Optimizer Sweep Comprehensive Dashboard',
                fontsize=18, fontweight='bold', y=0.995)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'comprehensive_dashboard.png'),
                bbox_inches='tight', dpi=300)
    print(f"Saved comprehensive dashboard to {output_dir}/comprehensive_dashboard.png")
    plt.close()


def create_optimizer_family_comparison(df: pd.DataFrame, output_dir: str):
    fig, axes = plt.subplots(1, 1, figsize=(16, 12))
    fig.suptitle('Optimizer Family Analysis', fontsize=18, fontweight='bold')

    df_valid = df[df['best_cer'] != float('inf')]
    
    # Heatmap: Optimizer vs Scheduler
    ax4 = axes
    pivot_data = df_valid.pivot_table(values='best_cer',
                                       index='optimizer',
                                       columns='scheduler',
                                       aggfunc='mean')

    if not pivot_data.empty:
        sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='RdYlGn_r', ax=ax4, cbar_kws={'label': 'Best CER'}, linewidths=0.5)
        ax4.set_title('Optimizer × Scheduler Performance Matrix', fontweight='bold')
        ax4.set_xlabel('Scheduler', fontweight='bold')
        ax4.set_ylabel('Optimizer', fontweight='bold')

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'optimizer_family_analysis.png'),
                bbox_inches='tight', dpi=300)
    print(f"Saved optimizer family analysis to {output_dir}/optimizer_family_analysis.png")
    plt.close()


def create_convergence_analysis(df: pd.DataFrame, output_dir: str):
    """
    Analyze convergence characteristics of different configurations.

    Args:
        df: DataFrame with all results
        output_dir: Directory to save plots
    """
    # Filter out runs from older setup (>30 steps)
    df = df[df['testCER'].apply(len) <= 30]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Convergence Analysis', fontsize=18, fontweight='bold')

    # 1. Top 10 CER curves
    ax1 = axes[0, 0]
    top10 = df.head(10)
    for idx, row in top10.iterrows():
        if len(row['testCER']) > 0:
            ax1.plot(row['testCER'], label=f"#{row['rank']} {row['optimizer']}", linewidth=2, alpha=0.7)
    ax1.set_xlabel('Evaluation Step')
    ax1.set_ylabel('Test CER')
    ax1.set_title('Top 10: Test CER Convergence', fontweight='bold')
    ax1.legend(fontsize=7, loc='best')
    ax1.grid(alpha=0.3)

    # 2. Top 10 Loss curves
    ax2 = axes[0, 1]
    for idx, row in top10.iterrows():
        if len(row['testLoss']) > 0:
            ax2.plot(row['testLoss'], label=f"#{row['rank']} {row['optimizer']}", linewidth=2, alpha=0.7)
    ax2.set_xlabel('Evaluation Step')
    ax2.set_ylabel('Test Loss')
    ax2.set_title('Top 10: Test Loss Convergence', fontweight='bold')
    ax2.legend(fontsize=7, loc='best')
    ax2.grid(alpha=0.3)

    # 3. Learning rate schedules
    ax3 = axes[0, 2]
    for idx, row in top10.iterrows():
        if len(row['learning_rates']) > 0:
            ax3.plot(row['learning_rates'], label=f"#{row['rank']} {row['scheduler']}", linewidth=2, alpha=0.7)
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Top 10: Learning Rate Schedules', fontweight='bold')
    ax3.set_yscale('log')
    ax3.legend(fontsize=7, loc='best')
    ax3.grid(alpha=0.3)

    # 4. Convergence speed comparison (by optimizer category)
    ax4 = axes[1, 0]
    for category in df['optimizer_category'].unique():
        category_runs = df[df['optimizer_category'] == category].head(3)
        cer_curves = []
        for _, row in category_runs.iterrows():
            if len(row['testCER']) > 0:
                cer_curves.append(row['testCER'])

        if cer_curves:
            # Average curves
            min_len = min(len(c) for c in cer_curves)
            avg_curve = np.mean([c[:min_len] for c in cer_curves], axis=0)
            ax4.plot(avg_curve, label=category, linewidth=2.5, alpha=0.8)

    ax4.set_xlabel('Evaluation Step')
    ax4.set_ylabel('Average Test CER')
    ax4.set_title('Convergence by Optimizer Category (Top 3 avg)', fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)

    # 5. Convergence speed comparison (by scheduler)
    ax5 = axes[1, 1]
    for scheduler in df['scheduler'].unique():
        scheduler_runs = df[df['scheduler'] == scheduler].head(3)
        cer_curves = []
        for _, row in scheduler_runs.iterrows():
            if len(row['testCER']) > 0:
                cer_curves.append(row['testCER'])

        if cer_curves:
            min_len = min(len(c) for c in cer_curves)
            avg_curve = np.mean([c[:min_len] for c in cer_curves], axis=0)
            ax5.plot(avg_curve, label=scheduler, linewidth=2.5, alpha=0.8)

    ax5.set_xlabel('Evaluation Step')
    ax5.set_ylabel('Average Test CER')
    ax5.set_title('Convergence by Scheduler (Top 3 avg)', fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3)

    # 6. Improvement over training
    ax6 = axes[1, 2]
    df_improvement = df[df['cer_improvement'] > 0].head(10)
    if len(df_improvement) > 0:
        colors = plt.cm.viridis(np.linspace(0, 1, len(df_improvement)))
        bars = ax6.barh(range(len(df_improvement)), df_improvement['cer_improvement_pct'], color=colors, edgecolor='black')
        ax6.set_yticks(range(len(df_improvement)))
        ax6.set_yticklabels([f"{row['optimizer']}" for _, row in df_improvement.iterrows()],
                           fontsize=9)
        ax6.set_xlabel('CER Improvement (%)')
        ax6.set_title('Top 10: CER Improvement During Training', fontweight='bold')
        ax6.invert_yaxis()
        ax6.grid(axis='x', alpha=0.3)

        # Add percentage labels
        for i, (bar, val) in enumerate(zip(bars, df_improvement['cer_improvement_pct'])):
            ax6.text(val, i, f' {val:.1f}%', va='center', fontsize=8)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'convergence_analysis.png'), bbox_inches='tight', dpi=300)
    print(f"Saved convergence analysis to {output_dir}/convergence_analysis.png")
    plt.close()


def create_hyperparameter_analysis(df: pd.DataFrame, output_dir: str):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Hyperparameter Sensitivity Analysis', fontsize=18, fontweight='bold')

    df_valid = df[df['best_cer'] != float('inf')]

    # 1. Learning Rate Distribution
    ax1 = axes[0, 0]
    df_lr = df_valid[df_valid['lr'].notna() & (df_valid['lr'] < 1.0)]
    if len(df_lr) > 0:
        lr_bins = np.logspace(np.log10(df_lr['lr'].min()), np.log10(df_lr['lr'].max()), 15)

        # Bin data
        df_lr['lr_bin'] = pd.cut(df_lr['lr'], bins=lr_bins)
        lr_stats = df_lr.groupby('lr_bin')['best_cer'].agg(['mean', 'std', 'count'])
        lr_stats = lr_stats[lr_stats['count'] > 0]

        bin_centers = [interval.mid for interval in lr_stats.index]
        ax1.errorbar(bin_centers, lr_stats['mean'], yerr=lr_stats['std'], fmt='o-', capsize=5, linewidth=2, markersize=8, alpha=0.7)
        ax1.set_xlabel('Learning Rate', fontweight='bold')
        ax1.set_ylabel('Mean Best CER', fontweight='bold')
        ax1.set_xscale('log')
        ax1.set_title('Learning Rate Sensitivity', fontweight='bold')
        ax1.grid(alpha=0.3)

    # 2. Weight Decay Distribution
    ax2 = axes[0, 1]
    df_wd = df_valid[df_valid['weight_decay'].notna()]
    if len(df_wd) > 0:
        wd_stats = df_wd.groupby('weight_decay')['best_cer'].agg(['mean', 'std', 'count'])

        x = range(len(wd_stats))
        ax2.bar(x, wd_stats['mean'], yerr=wd_stats['std'], capsize=5, alpha=0.7, color='teal', edgecolor='black')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'{wd:.0e}' for wd in wd_stats.index], rotation=45)
        ax2.set_ylabel('Mean Best CER', fontweight='bold')
        ax2.set_xlabel('Weight Decay', fontweight='bold')
        ax2.set_title('Weight Decay Sensitivity', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        # Add count labels
        for i, count in enumerate(wd_stats['count']):
            ax2.text(i, wd_stats['mean'].iloc[i], f'n={int(count)}', ha='center', va='bottom', fontsize=8)

    # 3. LR vs WD Heatmap
    ax3 = axes[1, 0]
    df_lr_wd = df_valid[(df_valid['lr'].notna()) &
                        (df_valid['weight_decay'].notna()) &
                        (df_valid['lr'] < 1.0)]

    if len(df_lr_wd) > 0:
        # Create bins for better visualization
        df_lr_wd['lr_bin'] = pd.qcut(df_lr_wd['lr'].rank(method='first'),
                                      q=min(5, len(df_lr_wd['lr'].unique())),
                                      duplicates='drop')

        pivot = df_lr_wd.pivot_table(values='best_cer',
                                     index='weight_decay',
                                     columns='lr_bin',
                                     aggfunc='mean')

        if not pivot.empty:
            sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn_r',
                       ax=ax3, cbar_kws={'label': 'Best CER'}, linewidths=0.5)
            ax3.set_title('LR × Weight Decay Interaction', fontweight='bold')
            ax3.set_xlabel('Learning Rate (binned)', fontweight='bold')
            ax3.set_ylabel('Weight Decay', fontweight='bold')

    # 4. Performance distribution by rank
    ax4 = axes[1, 1]
    rank_bins = [1, 6, 11, 21, len(df_valid)+1]
    rank_labels = ['Top 5', 'Top 6-10', 'Top 11-20', 'Rest']
    df_valid['rank_category'] = pd.cut(df_valid['rank'], bins=rank_bins, labels=rank_labels, right=False)

    rank_data = [df_valid[df_valid['rank_category'] == cat]['best_cer'].values
                for cat in rank_labels if cat in df_valid['rank_category'].values]

    bp = ax4.boxplot(rank_data, tick_labels=rank_labels[:len(rank_data)],
                    patch_artist=True, showmeans=True)

    colors = plt.cm.RdYlGn(np.linspace(0.7, 0.3, len(rank_data)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax4.set_ylabel('Best CER', fontweight='bold')
    ax4.set_xlabel('Rank Category', fontweight='bold')
    ax4.set_title('Performance Distribution by Rank', fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'hyperparameter_analysis.png'),
                bbox_inches='tight', dpi=300)
    print(f"Saved hyperparameter analysis to {output_dir}/hyperparameter_analysis.png")
    plt.close()


def generate_summary_report(df: pd.DataFrame, output_dir: str):
    """
    Generate comprehensive text and CSV summary reports.

    Args:
        df: DataFrame with all results
        output_dir: Directory to save reports
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save full results CSV
    csv_path = os.path.join(output_dir, 'full_results.csv')
    df_export = df.drop(columns=['testCER', 'testLoss', 'trainLoss', 'learning_rates'])
    df_export.to_csv(csv_path, index=False)
    print(f"Saved full results to {csv_path}")

    # Generate text report
    report_lines = [
        "=" * 100,
        "OPTIMIZER SWEEP RESULTS - COMPREHENSIVE ANALYSIS",
        "=" * 100,
        f"\nGenerated: {pd.Timestamp.now()}",
        f"Total Configurations: {len(df)}",
        f"Valid Configurations: {len(df[df['best_cer'] != float('inf')])}",
        "\n" + "=" * 100,
        "TOP 10 CONFIGURATIONS",
        "=" * 100,
    ]

    top10 = df.head(10)[['rank', 'optimizer', 'scheduler', 'lr', 'weight_decay',
                         'best_cer', 'final_cer', 'cer_improvement_pct', 'total_time_seconds']]
    report_lines.append("\n" + top10.to_string(index=False))

    report_lines.extend([
        "\n" + "=" * 100,
        "OPTIMIZER CATEGORY STATISTICS",
        "=" * 100,
    ])

    df_valid = df[df['best_cer'] != float('inf')]
    if len(df_valid) > 0:
        cat_stats = df_valid.groupby('optimizer_category')['best_cer'].agg([
            ('count', 'count'),
            ('mean', 'mean'),
            ('std', 'std'),
            ('min', 'min'),
            ('max', 'max')
        ]).sort_values('mean')
        report_lines.append("\n" + cat_stats.to_string())

        report_lines.extend([
            "\n" + "=" * 100,
            "OPTIMIZER STATISTICS (Individual)",
            "=" * 100,
        ])

        opt_stats = df_valid.groupby('optimizer')['best_cer'].agg([
            ('count', 'count'),
            ('mean', 'mean'),
            ('std', 'std'),
            ('min', 'min')
        ]).sort_values('mean').head(15)
        report_lines.append("\n" + opt_stats.to_string())

        report_lines.extend([
            "\n" + "=" * 100,
            "SCHEDULER STATISTICS",
            "=" * 100,
        ])

        sched_stats = df_valid.groupby('scheduler')['best_cer'].agg([
            ('count', 'count'),
            ('mean', 'mean'),
            ('std', 'std'),
            ('min', 'min')
        ]).sort_values('mean')
        report_lines.append("\n" + sched_stats.to_string())

        report_lines.extend([
            "\n" + "=" * 100,
            "KEY FINDINGS",
            "=" * 100,
        ])

        best = df.iloc[0]
        report_lines.extend([
            f"\nBest Overall Configuration:",
            f"  - Optimizer: {best['optimizer']} ({best['optimizer_category']})",
            f"  - Scheduler: {best['scheduler']}",
            f"  - Learning Rate: {best['lr']}",
            f"  - Weight Decay: {best['weight_decay']}",
            f"  - Best CER: {best['best_cer']:.6f}",
            f"  - Final CER: {best['final_cer']:.6f}",
            f"  - Improvement: {best['cer_improvement_pct']:.2f}%",
            f"  - Training Time: {best['total_time_seconds']:.1f}s",
        ])

        # Best by category
        report_lines.append(f"\nBest Configuration by Category:")
        for category in df_valid['optimizer_category'].unique():
            cat_best = df_valid[df_valid['optimizer_category'] == category].iloc[0]
            report_lines.append(
                f"  - {category}: {cat_best['optimizer']} + {cat_best['scheduler']} "
                f"(CER: {cat_best['best_cer']:.6f})"
            )

        # Adaptive vs Non-adaptive
        if 'is_adaptive' in df_valid.columns:
            adaptive_mean = df_valid[df_valid['is_adaptive']]['best_cer'].mean()
            nonadaptive_mean = df_valid[~df_valid['is_adaptive']]['best_cer'].mean()
            report_lines.extend([
                f"\nAdaptive vs Non-Adaptive Optimizers:",
                f"  - Adaptive mean CER: {adaptive_mean:.6f}",
                f"  - Non-adaptive mean CER: {nonadaptive_mean:.6f}",
            ])

    report_lines.append("\n" + "=" * 100)

    # Save report
    report_path = os.path.join(output_dir, 'comprehensive_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"Saved comprehensive report to {report_path}")
    print("\n" + '\n'.join(report_lines))


def main(sweep_dir: str = '/home/iseanbhanot/optimizer_sweep_results',
         output_dir: str = None):
    """
    Main function to generate all visualizations and reports.

    Args:
        sweep_dir: Directory containing sweep results
        output_dir: Directory to save outputs (default: sweep_dir/visualizations)
    """
    if output_dir is None:
        output_dir = os.path.join(sweep_dir, 'visualizations')

    print("\n" + "=" * 100)
    print("ENHANCED OPTIMIZER SWEEP VISUALIZATION")
    print("=" * 100 + "\n")

    # Load data
    print("Loading results...")
    df = load_all_results(sweep_dir)
    print(f"Loaded {len(df)} configurations\n")

    if len(df) == 0:
        print("No results found!")
        return

    # Generate visualizations
    print("Generating comprehensive dashboard...")
    create_comprehensive_dashboard(df, output_dir)

    print("Generating optimizer family comparison...")
    create_optimizer_family_comparison(df, output_dir)

    print("Generating convergence analysis...")
    create_convergence_analysis(df, output_dir)

    print("Generating hyperparameter analysis...")
    create_hyperparameter_analysis(df, output_dir)

    print("Generating summary reports...")
    generate_summary_report(df, output_dir)

    print("\n" + "=" * 100)
    print(f"ALL VISUALIZATIONS COMPLETE!")
    print(f"Output directory: {output_dir}")
    print("=" * 100 + "\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate enhanced visualizations for optimizer sweep results'
    )
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
        help='Directory to save visualizations (default: sweep_dir/visualizations)'
    )

    args = parser.parse_args()
    main(sweep_dir=args.sweep_dir, output_dir=args.output_dir)
