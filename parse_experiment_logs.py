import os
import re
import glob
import pandas as pd

def parse_log(file_path):
    """Parses a RecBole training log file into epoch history and final results."""
    epochs = []
    best_valid = {}
    test_result = {}
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
    epoch_pat = re.compile(r'epoch\s+(\d+)\s+training\s+\[time:\s*([\d\.]+)s,\s*train loss:\s*([\d\.]+)\]')
    eval_pat = re.compile(r'epoch\s+(\d+)\s+evaluating\s+\[time:\s*([\d\.]+)s,\s*valid_score:\s*([\d\.]+)\]')
    valid_metrics_pat = re.compile(r'recall@20\s*:\s*([\d\.]+)\s+recall@50\s*:\s*([\d\.]+)\s+ndcg@20\s*:\s*([\d\.]+)\s+ndcg@50\s*:\s*([\d\.]+)\s+hit@20\s*:\s*([\d\.]+)\s+hit@50\s*:\s*([\d\.]+)\s+precision@20\s*:\s*([\d\.]+)\s+precision@50\s*:\s*([\d\.]+)\s+mrr@20\s*:\s*([\d\.]+)\s+mrr@50\s*:\s*([\d\.]+)')
    
    current_epoch = None
    train_time = 0.0
    train_loss = 0.0
    eval_time = 0.0
    valid_score = 0.0
    
    for i, line in enumerate(lines):
        m_ep = epoch_pat.search(line)
        if m_ep:
            current_epoch = int(m_ep.group(1))
            train_time = float(m_ep.group(2))
            train_loss = float(m_ep.group(3))
            continue
            
        m_ev = eval_pat.search(line)
        if m_ev:
            eval_time = float(m_ev.group(2))
            valid_score = float(m_ev.group(3))
            
            # Next line usually has valid result
            if i + 2 < len(lines):
                m_met = valid_metrics_pat.search(lines[i+2])
                if m_met:
                    epochs.append({
                        'epoch': current_epoch,
                        'train_time_s': train_time,
                        'train_loss': train_loss,
                        'eval_time_s': eval_time,
                        'recall@20': float(m_met.group(1)),
                        'recall@50': float(m_met.group(2)),
                        'ndcg@20': float(m_met.group(3)),
                        'ndcg@50': float(m_met.group(4)),
                        'hit@20': float(m_met.group(5)),
                        'hit@50': float(m_met.group(6)),
                        'precision@20': float(m_met.group(7)),
                        'precision@50': float(m_met.group(8)),
                        'mrr@20': float(m_met.group(9)),
                        'mrr@50': float(m_met.group(10))
                    })
                    
        if 'test result:' in line:
            # Parse test metrics dictionary
            test_line = line.strip().split('test result:')[-1]
            # Match recall@20, ndcg@20, etc.
            pairs = re.findall(r"'([\w@]+)':\s*np\.float64\(([\d\.]+)\)", test_line)
            if not pairs:
                pairs = re.findall(r"\('([\w@]+)',\s*([\d\.]+)\)", test_line)
            if not pairs:
                pairs = re.findall(r"'([\w@]+)':\s*([\d\.]+)", test_line)
            for k, v in pairs:
                test_result[k] = float(v)

    df_epochs = pd.DataFrame(epochs)
    return df_epochs, test_result

if __name__ == '__main__':
    log_files = glob.glob('log/SGCL/*.log')
    print(f"Found {len(log_files)} logs in log/SGCL/")
    for lf in sorted(log_files, key=os.path.getmtime, reverse=True)[:5]:
        print(f"File: {os.path.basename(lf)} (mtime: {os.path.getmtime(lf)})")
