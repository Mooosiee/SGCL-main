"""
Controlled Benchmark Runner for FS-SGCL vs. Graph Recommender Baselines.

Supported models:
  - LightGCN (SIGIR 2020, BPR + LightGCN)
  - SGL      (SIGIR 2021, BPR + Graph Augmentation InfoNCE)
  - SimGCL   (SIGIR 2022, BPR + Noise Perturbation InfoNCE)
  - LightGCL (ICLR 2023,  BPR + SVD-Guided InfoNCE)
  - SGCL     (RecSys 2025, Vanilla Supervised GCL Baseline)
  - FS-SGCL  (Ours, Full-Spectrum Diagnosis-Driven SGCL)

All models share:
  - Exact same dataset, 5-core filtering ([5, inf)), and 8:1:1 RS user-grouped split per seed
  - Same embedding_size=64, batch_size=1024, Adam optimizer, stopping_step=10 on valid NDCG@20
  - Same full-sort catalog ranking protocol over Recall@20/50, NDCG@20/50, Hit@20, MRR@20

Example (Fast Single-Seed Pilot on Toys & Games):
  python run_benchmark.py --models LightGCN SGL SimGCL SGCL FS-SGCL --datasets amazon-toys-games --seeds 2020

Example (Full Multi-Seed + 4-Trial Validation Tuning on GPU / Kaggle):
  python run_benchmark.py --models LightGCN SGL SimGCL SGCL FS-SGCL --datasets amazon-beauty amazon-toys-games --seeds 2020 2021 2022 --tune
"""

import os
import sys
import time
import json
import argparse
import warnings
import pandas as pd
import numpy as np
import torch
from logging import getLogger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recbole.utils import init_seed
from recbole_gnn.config import Config
from recbole_gnn.logger import init_logger
from recbole_gnn.utils import create_dataset, data_preparation, get_model, get_trainer


# Native / published hyperparameter defaults (fair baseline starting points)
DEFAULT_MODEL_CONFIGS = {
    'LightGCN': {
        'base_model': 'LightGCN',
        'overrides': {
            'embedding_size': 64,
            'n_layers': 3,
            'learning_rate': 0.001,
            'reg_weight': 1e-5,
            'require_pow': True,
            'weight_decay': 0.0,
        },
        'tune_grid': [
            {'reg_weight': 1e-5, 'learning_rate': 0.001},
            {'reg_weight': 1e-4, 'learning_rate': 0.001},
            {'reg_weight': 1e-3, 'learning_rate': 0.001},
            {'reg_weight': 1e-4, 'learning_rate': 0.002},
        ],
    },
    'SGL': {
        'base_model': 'SGL',
        'overrides': {
            'embedding_size': 64,
            'n_layers': 3,
            'learning_rate': 0.001,
            'type': 'ED',
            'reg_weight': 1e-5,
            'ssl_tau': 0.5,
            'ssl_weight': 0.05,
            'drop_ratio': 0.1,
            'lamda': 1.0,
            'weight_decay': 0.0,
        },
        'tune_grid': [
            {'ssl_weight': 0.05, 'drop_ratio': 0.1, 'ssl_tau': 0.5},
            {'ssl_weight': 0.10, 'drop_ratio': 0.1, 'ssl_tau': 0.2},
            {'ssl_weight': 0.05, 'drop_ratio': 0.2, 'ssl_tau': 0.2},
            {'ssl_weight': 0.20, 'drop_ratio': 0.1, 'ssl_tau': 0.2},
        ],
    },
    'SimGCL': {
        'base_model': 'SimGCL',
        'overrides': {
            'embedding_size': 64,
            'n_layers': 3,
            'learning_rate': 0.001,
            'reg_weight': 1e-4,
            'require_pow': False,
            'lambda': 0.2,
            'eps': 0.1,
            'temperature': 0.2,
            'weight_decay': 0.0,
        },
        'tune_grid': [
            {'lambda': 0.2, 'eps': 0.1, 'reg_weight': 1e-4},
            {'lambda': 0.5, 'eps': 0.1, 'reg_weight': 1e-4},
            {'lambda': 0.2, 'eps': 0.2, 'reg_weight': 1e-4},
            {'lambda': 0.2, 'eps': 0.1, 'reg_weight': 1e-5},
        ],
    },
    'LightGCL': {
        'base_model': 'LightGCL',
        'overrides': {
            'embedding_size': 64,
            'n_layers': 3,
            'learning_rate': 0.001,
            'dropout': 0.0,
            'temp': 0.2,
            'lambda1': 0.2,
            'lambda2': 1e-7,
            'q': 5,
            'weight_decay': 0.0,
        },
        'tune_grid': [
            {'lambda1': 0.2, 'temp': 0.2},
            {'lambda1': 0.5, 'temp': 0.2},
            {'lambda1': 0.2, 'temp': 0.5},
            {'lambda1': 0.1, 'temp': 0.2},
        ],
    },
    'SGCL': {
        'base_model': 'SGCL',
        'overrides': {
            'embedding_size': 64,
            'n_layers': 3,
            'learning_rate': 0.001,
            'weight_decay': 1e-6,
            'temperature': 0.2,
            'mask_mode': 'none',
            'pop_beta': 0.0,
            'pop_alpha': 0.0,
            'cross_weight': 0.0,
            'layer_weight_mode': 'mean',
        },
        'tune_grid': [
            {'temperature': 0.2},
            {'temperature': 0.1},
            {'temperature': 0.5},
            {'temperature': 0.2, 'learning_rate': 0.002},
        ],
    },
    'FS-SGCL': {
        'base_model': 'SGCL',
        'overrides': {
            'embedding_size': 64,
            'n_layers': 3,
            'learning_rate': 0.001,
            'weight_decay': 1e-6,
            'temperature': 0.2,
            'mask_mode': 'all',
            'pop_beta': 0.5,
            'pop_alpha': 0.0,
            'cross_weight': 0.2,
            'layer_weight_mode': 'decay',
        },
        'tune_grid': [
            {'pop_beta': 0.5, 'cross_weight': 0.2},
            {'pop_beta': 0.5, 'cross_weight': 0.1},
            {'pop_beta': 0.3, 'cross_weight': 0.2},
            {'pop_beta': 0.5, 'cross_weight': 0.5},
        ],
    },
}


def run_single_experiment(model_label, dataset_name, seed, extra_overrides=None, epochs=None, gpu='0', checkpoint_dir='saved/benchmark'):
    spec = DEFAULT_MODEL_CONFIGS[model_label]
    base_model = spec['base_model']

    cfg_dict = dict(spec['overrides'])
    if extra_overrides:
        cfg_dict.update(extra_overrides)

    os.makedirs(checkpoint_dir, exist_ok=True)
    cfg_dict.update({
        'seed': int(seed),
        'reproducibility': True,
        'save_dataloaders': False,   # Prevents cross-model dataloader cache mismatches
        'save_dataset': False,
        'show_progress': False,
        'checkpoint_dir': checkpoint_dir,
    })
    if epochs is not None:
        cfg_dict['epochs'] = int(epochs)

    config = Config(
        model=base_model,
        dataset=dataset_name,
        gpu=gpu,
        config_file_list=['config.yaml'],
        config_dict=cfg_dict,
    )
    init_seed(config['seed'], config['reproducibility'])
    suffix = f"bench_{model_label}_{dataset_name}_s{seed}"
    init_logger(config, suffix)
    logger = getLogger()

    logger.info(f"=== BENCHMARK RUN: {model_label} | Dataset: {dataset_name} | Seed: {seed} | Device: {config['device']} ===")
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config['seed'], config['reproducibility'])
    model = get_model(base_model)(config, train_data.dataset).to(config['device'])
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    trainer = get_trainer(config['MODEL_TYPE'], base_model)(config, model)

    t0 = time.time()
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )
    train_wall_s = time.time() - t0
    best_epoch = len(trainer.train_loss_dict) - trainer.cur_step - 1
    total_epochs = len(trainer.train_loss_dict)

    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=False)

    logger.info(f"[{model_label} | {dataset_name} | seed={seed}] Best valid NDCG@20={best_valid_score:.4f} (ep {best_epoch}/{total_epochs}, {train_wall_s:.1f}s)")
    logger.info(f"[{model_label} | {dataset_name} | seed={seed}] Test: {test_result}")

    # Close logger handlers cleanly to avoid duplicate lines across runs
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()

    return {
        'dataset': dataset_name,
        'model': model_label,
        'seed': int(seed),
        'device': str(config['device']),
        'params': int(n_params),
        'best_epoch': int(max(best_epoch, 0)),
        'total_epochs': int(total_epochs),
        'train_time_s': round(train_wall_s, 2),
        'best_valid_ndcg@20': float(best_valid_score),
        'recall@20': float(test_result.get('recall@20', np.nan)),
        'ndcg@20': float(test_result.get('ndcg@20', np.nan)),
        'recall@50': float(test_result.get('recall@50', np.nan)),
        'ndcg@50': float(test_result.get('ndcg@50', np.nan)),
        'hit@20': float(test_result.get('hit@20', np.nan)),
        'mrr@20': float(test_result.get('mrr@20', np.nan)),
        'hparams': json.dumps(extra_overrides if extra_overrides else {}),
        'checkpoint_path': getattr(trainer, 'saved_model_file', ''),
    }


def generate_summary_tables(runs_csv, summary_csv, tables_md):
    if not os.path.exists(runs_csv):
        return
    df = pd.read_csv(runs_csv)
    if df.empty:
        return

    metrics = ['recall@20', 'ndcg@20', 'recall@50', 'ndcg@50', 'hit@20', 'mrr@20', 'train_time_s', 'best_epoch']
    grouped_rows = []
    for (dataset, model), g in df.groupby(['dataset', 'model'], sort=False):
        row = {
            'dataset': dataset,
            'model': model,
            'n_seeds': len(g),
            'params': int(g['params'].iloc[0]),
        }
        for m in metrics:
            mean_v = g[m].mean()
            std_v = g[m].std(ddof=1) if len(g) > 1 else 0.0
            row[f'{m}_mean'] = mean_v
            row[f'{m}_std'] = std_v
            if m in ['train_time_s', 'best_epoch']:
                row[m] = f"{mean_v:.1f}" if len(g) == 1 else f"{mean_v:.1f} ± {std_v:.1f}"
            else:
                row[m] = f"{mean_v:.4f}" if len(g) == 1 else f"{mean_v:.4f} ± {std_v:.4f}"
        grouped_rows.append(row)

    summary_df = pd.DataFrame(grouped_rows)
    summary_df.to_csv(summary_csv, index=False)

    md_lines = ["# Benchmark Comparison Summary\n"]
    for dataset in df['dataset'].unique():
        sub = summary_df[summary_df['dataset'] == dataset]
        md_lines.append(f"## Dataset: {dataset}\n")
        md_lines.append("| Model | Seeds | Recall@20 | NDCG@20 | Recall@50 | NDCG@50 | Hit@20 | MRR@20 |")
        md_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for _, r in sub.iterrows():
            md_lines.append(
                f"| {r['model']} | {r['n_seeds']} | {r['recall@20']} | {r['ndcg@20']} | {r['recall@50']} | {r['ndcg@50']} | {r['hit@20']} | {r['mrr@20']} |"
            )
        md_lines.append("")

        md_lines.append(f"### Efficiency — {dataset}\n")
        md_lines.append("| Model | Params | Best Epoch | Wall Time (s) |")
        md_lines.append("| :--- | :--- | :--- | :--- |")
        for _, r in sub.iterrows():
            md_lines.append(f"| {r['model']} | {r['params']:,} | {r['best_epoch']} | {r['train_time_s']} |")
        md_lines.append("\n")

    with open(tables_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    print('\n'.join(md_lines))


def main():
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(description='Controlled Benchmark Runner for SGCL & Baselines')
    parser.add_argument(
        '--models', nargs='+',
        default=['LightGCN', 'SGL', 'SimGCL', 'SGCL', 'FS-SGCL'],
        choices=list(DEFAULT_MODEL_CONFIGS.keys()),
        help='Models to benchmark'
    )
    parser.add_argument(
        '--datasets', nargs='+',
        default=['amazon-toys-games', 'amazon-beauty'],
        help='Datasets to evaluate on'
    )
    parser.add_argument('--seeds', nargs='+', type=int, default=[2020], help='Random seeds (e.g. 2020 2021 2022)')
    parser.add_argument('--tune', action='store_true', help='Run 4-trial validation tuning on first seed before multi-seed runs')
    parser.add_argument('--epochs', type=int, default=None, help='Optional epoch cap (default: 300 with stopping_step=10)')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID if CUDA is available')
    parser.add_argument('--out_dir', type=str, default='results/benchmark', help='Output directory for CSVs and summary tables')
    parser.add_argument('--no_skip', action='store_true', help='Do not skip already-completed (dataset, model, seed) runs')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    runs_csv = os.path.join(args.out_dir, 'benchmark_runs.csv')
    tune_csv = os.path.join(args.out_dir, 'tuning_trials.csv')
    summary_csv = os.path.join(args.out_dir, 'benchmark_summary_mean_std.csv')
    tables_md = os.path.join(args.out_dir, 'benchmark_tables.md')

    completed = set()
    all_rows = []
    if os.path.exists(runs_csv) and not args.no_skip:
        existing_df = pd.read_csv(runs_csv)
        all_rows = existing_df.to_dict('records')
        for r in all_rows:
            completed.add((r['dataset'], r['model'], int(r['seed'])))
        print(f"Loaded {len(completed)} completed benchmark runs from {runs_csv}")

    tune_rows = []
    if os.path.exists(tune_csv):
        tune_rows = pd.read_csv(tune_csv).to_dict('records')

    for dataset in args.datasets:
        for model_label in args.models:
            best_overrides = None

            # Check if all seeds for this (dataset, model) are already done
            if not args.no_skip and all((dataset, model_label, int(s)) in completed for s in args.seeds):
                print(f"Skipping all seeds for already-completed ({dataset}, {model_label})")
                continue

            # Optional fair 4-trial validation search selected purely on valid NDCG@20
            if args.tune:
                grid = DEFAULT_MODEL_CONFIGS[model_label]['tune_grid']
                best_valid = -1.0
                tune_seed = args.seeds[0]
                print(f"\n--- Validation Tuning [{model_label} on {dataset}] ({len(grid)} trials, seed={tune_seed}) ---")
                for t_idx, trial_ov in enumerate(grid):
                    res = run_single_experiment(
                        model_label, dataset, tune_seed,
                        extra_overrides=trial_ov, epochs=args.epochs, gpu=args.gpu,
                        checkpoint_dir=os.path.join('saved/benchmark', f'tune_{model_label}_{dataset}')
                    )
                    res['trial_idx'] = t_idx
                    tune_rows.append(res)
                    pd.DataFrame(tune_rows).to_csv(tune_csv, index=False)
                    if res['best_valid_ndcg@20'] > best_valid:
                        best_valid = res['best_valid_ndcg@20']
                        best_overrides = trial_ov
                        # If tune_seed was already tested with winning config, reuse result cleanly
                        winning_first_res = dict(res)
                print(f"Winner for [{model_label} | {dataset}]: {best_overrides} (valid NDCG@20={best_valid:.4f})")

                if (dataset, model_label, tune_seed) not in completed:
                    winning_first_res.pop('trial_idx', None)
                    all_rows.append(winning_first_res)
                    completed.add((dataset, model_label, tune_seed))
                    pd.DataFrame(all_rows).to_csv(runs_csv, index=False)

            for seed in args.seeds:
                if (dataset, model_label, int(seed)) in completed:
                    print(f"Skipping already completed: ({dataset}, {model_label}, seed={seed})")
                    continue

                print(f"\n>>> Running [{model_label}] on [{dataset}] (seed={seed})...")
                res = run_single_experiment(
                    model_label, dataset, seed,
                    extra_overrides=best_overrides, epochs=args.epochs, gpu=args.gpu
                )
                all_rows.append(res)
                completed.add((dataset, model_label, int(seed)))
                pd.DataFrame(all_rows).to_csv(runs_csv, index=False)

    generate_summary_tables(runs_csv, summary_csv, tables_md)


if __name__ == '__main__':
    main()
