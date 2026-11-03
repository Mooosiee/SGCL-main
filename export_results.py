import os
import shutil
import re
import pandas as pd
from parse_experiment_logs import parse_log

def main():
    os.makedirs('results/logs', exist_ok=True)
    
    runs = [
        # Beauty
        ('amazon-beauty', 'Baseline', 'log/SGCL/SGCL-amazon-beauty-None-Oct-25-2026_21-32-18-96337f.log', 'amazon-beauty_baseline.log', 'amazon-beauty_baseline_per_epoch.csv'),
        ('amazon-beauty', 'FN-SGCL', 'log/SGCL/SGCL-amazon-beauty-FN_SGCL_resumed-Oct-29-2026_01-11-31-bbbc38.log', 'amazon-beauty_fn_sgcl.log', 'amazon-beauty_fn_sgcl_per_epoch.csv'),
        ('amazon-beauty', 'Pop-FN-SGCL', 'log/SGCL/SGCL-amazon-beauty-Pop_FN_SGCL_beta05-Oct-31-2026_19-26-08-092ad3.log', 'amazon-beauty_pop_fn_sgcl.log', 'amazon-beauty_pop_fn_sgcl_per_epoch.csv'),
        ('amazon-beauty', 'FS-SGCL', 'log/SGCL/SGCL-amazon-beauty-FS_SGCL_cw02-Nov-01-2026_13-27-36-89899c.log', 'amazon-beauty_fs_sgcl.log', 'amazon-beauty_fs_sgcl_per_epoch.csv'),
        
        # Toys & Games
        ('amazon-toys-games', 'Baseline', 'log/SGCL/SGCL-amazon-toys-games-None-Oct-25-2026_22-25-58-1a4359.log', 'amazon-toys-games_baseline.log', 'amazon-toys-games_baseline_per_epoch.csv'),
        ('amazon-toys-games', 'FN-SGCL', 'log/SGCL/SGCL-amazon-toys-games-FN_SGCL-Oct-29-2026_11-12-31-42873d.log', 'amazon-toys-games_fn_sgcl.log', 'amazon-toys-games_fn_sgcl_per_epoch.csv'),
        ('amazon-toys-games', 'Pop-FN-SGCL', 'log/SGCL/SGCL-amazon-toys-games-Pop_FN_SGCL_resumed-Nov-01-2026_08-51-16-c6e2cd.log', 'amazon-toys-games_pop_fn_sgcl.log', 'amazon-toys-games_pop_fn_sgcl_per_epoch.csv'),
        ('amazon-toys-games', 'FS-SGCL', 'log/SGCL/SGCL-amazon-toys-games-FS_SGCL_cw02-Nov-01-2026_12-22-24-98b995.log', 'amazon-toys-games_fs_sgcl.log', 'amazon-toys-games_fs_sgcl_per_epoch.csv'),
    ]
    
    summary_rows = []
    
    for dataset, variant, log_path, dest_log, dest_csv in runs:
        if not os.path.exists(log_path):
            print(f"Warning: {log_path} not found.")
            continue
            
        # 1. Copy log to results/logs/
        shutil.copyfile(log_path, os.path.join('results/logs', dest_log))
        
        # 2. Parse log
        df_epochs, test_result = parse_log(log_path)
        
        # Save per-epoch CSV
        out_csv_path = os.path.join('results', dest_csv)
        df_epochs.to_csv(out_csv_path, index=False)
        print(f"Exported {dataset} [{variant}] -> {len(df_epochs)} epochs saved to {out_csv_path}")
        
        # Summary row
        row = {'Dataset': dataset, 'Variant': variant}
        row.update(test_result)
        summary_rows.append(row)
        
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv('results/all_variants_test_metrics.csv', index=False)
    print("\n--- Summary of All Variants Test Metrics ---")
    print(df_summary[['Dataset', 'Variant', 'recall@20', 'ndcg@20', 'recall@50', 'ndcg@50']].to_string(index=False))

if __name__ == '__main__':
    main()
