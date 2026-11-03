import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from recbole_gnn.config import Config
from recbole_gnn.utils import create_dataset, data_preparation, get_model

def evaluate_popularity_strata(dataset_name, baseline_ckpt, fssgcl_ckpt, output_dir='results'):
    print(f"\n==========================================")
    print(f" Popularity Stratification Analysis: {dataset_name}")
    print(f"==========================================")
    
    os.makedirs(output_dir, exist_ok=True)
    config = Config(model='SGCL', dataset=dataset_name, config_file_list=['config.yaml'])
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    num_users = dataset.user_num
    num_items = dataset.item_num
    
    # 1. Compute item degrees on train set
    inter_feat = train_data.dataset.inter_feat
    uid_col = train_data.dataset.uid_field
    iid_col = train_data.dataset.iid_field
    
    train_uids = inter_feat[uid_col].numpy()
    train_iids = inter_feat[iid_col].numpy()
    
    item_deg = np.zeros(num_items, dtype=np.int64)
    unique_items, counts = np.unique(train_iids, return_counts=True)
    item_deg[unique_items] = counts
    
    # History mask: list of train items per user
    user_train_history = [set() for _ in range(num_users)]
    for u, i in zip(train_uids, train_iids):
        user_train_history[u].add(i)
        
    # Also mask valid items so evaluation strictly matches test protocol
    valid_feat = valid_data.dataset.inter_feat
    for u, i in zip(valid_feat[uid_col].numpy(), valid_feat[iid_col].numpy()):
        user_train_history[u].add(i)
        
    # 2. Popularity Buckets based on training frequency
    # We partition non-padding items (index 1 to num_items-1)
    all_item_degrees = item_deg[1:]
    q50 = np.percentile(all_item_degrees, 50)
    q80 = np.percentile(all_item_degrees, 80)
    
    head_items = set(np.where(item_deg >= q80)[0]) - {0}
    mid_items = set(np.where((item_deg >= q50) & (item_deg < q80))[0]) - {0}
    tail_items = set(np.where((item_deg < q50) & (item_deg > 0))[0]) - {0}
    
    print(f"Item Strata Counts: Head (>={q80:.0f} deg): {len(head_items)}, Mid ({q50:.0f}-{q80:.0f} deg): {len(mid_items)}, Tail (<{q50:.0f} deg): {len(tail_items)}")
    
    # 3. Test Ground Truth per user
    test_feat = test_data.dataset.inter_feat
    test_uids = test_feat[uid_col].numpy()
    test_iids = test_feat[iid_col].numpy()
    
    user_test_items = {}
    for u, i in zip(test_uids, test_iids):
        if u not in user_test_items:
            user_test_items[u] = []
        user_test_items[u].append(i)
        
    eval_users = sorted(list(user_test_items.keys()))
    print(f"Total test users: {len(eval_users)}, Total test interactions: {len(test_iids)}")
    
    # Function to evaluate a model checkpoint
    def evaluate_model(ckpt_path, model_name):
        print(f"\nEvaluating {model_name} from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model = get_model('SGCL')(config, train_data.dataset)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        
        with torch.no_grad():
            u_e, i_e = model.forward()
            
        # Metrics collectors
        ndcg_all = []
        ndcg_head = []
        ndcg_mid = []
        ndcg_tail = []
        
        rec_share_head = 0
        rec_share_mid = 0
        rec_share_tail = 0
        total_recs = 0
        
        batch_size = 1024
        for b_start in range(0, len(eval_users), batch_size):
            b_users = eval_users[b_start : b_start + batch_size]
            b_u_tensor = torch.tensor(b_users, dtype=torch.long)
            b_u_emb = u_e[b_u_tensor] # (B, dim)
            
            # Scores for all items
            scores = torch.matmul(b_u_emb, i_e.transpose(0, 1)).numpy() # (B, num_items)
            scores[:, 0] = -1e9 # Mask padding item
            
            for idx, u in enumerate(b_users):
                # Mask train + valid history
                if user_train_history[u]:
                    scores[idx, list(user_train_history[u])] = -1e9
                    
                # Top 20 items
                topk_items = np.argpartition(scores[idx], -20)[-20:]
                topk_items = topk_items[np.argsort(-scores[idx][topk_items])]
                
                # Recommendation shares
                for item in topk_items:
                    if item in head_items:
                        rec_share_head += 1
                    elif item in mid_items:
                        rec_share_mid += 1
                    elif item in tail_items:
                        rec_share_tail += 1
                total_recs += 20
                
                gt_items = set(user_test_items[u])
                gt_head = gt_items.intersection(head_items)
                gt_mid = gt_items.intersection(mid_items)
                gt_tail = gt_items.intersection(tail_items)
                
                # Helper for NDCG@20
                def compute_ndcg(top_list, target_set):
                    if not target_set:
                        return None
                    dcg = 0.0
                    for rank, item in enumerate(top_list):
                        if item in target_set:
                            dcg += 1.0 / np.log2(rank + 2)
                    idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(target_set), 20)))
                    return dcg / idcg if idcg > 0 else 0.0
                
                nd_all = compute_ndcg(topk_items, gt_items)
                if nd_all is not None:
                    ndcg_all.append(nd_all)
                    
                nd_h = compute_ndcg(topk_items, gt_head)
                if nd_h is not None:
                    ndcg_head.append(nd_h)
                    
                nd_m = compute_ndcg(topk_items, gt_mid)
                if nd_m is not None:
                    ndcg_mid.append(nd_m)
                    
                nd_t = compute_ndcg(topk_items, gt_tail)
                if nd_t is not None:
                    ndcg_tail.append(nd_t)
                    
        return {
            'model': model_name,
            'NDCG@20 (Overall)': np.mean(ndcg_all),
            'NDCG@20 (Head)': np.mean(ndcg_head),
            'NDCG@20 (Mid)': np.mean(ndcg_mid),
            'NDCG@20 (Tail)': np.mean(ndcg_tail),
            'Rec Share Head (%)': (rec_share_head / total_recs) * 100,
            'Rec Share Mid (%)': (rec_share_mid / total_recs) * 100,
            'Rec Share Tail (%)': (rec_share_tail / total_recs) * 100,
        }
        
    res_base = evaluate_model(baseline_ckpt, "Vanilla SGCL (Baseline)")
    res_fs = evaluate_model(fssgcl_ckpt, "FS-SGCL (Ours)")
    
    df_res = pd.DataFrame([res_base, res_fs])
    print("\n--- Popularity Stratification Results ---")
    print(df_res.to_string(index=False))
    
    # Save CSV
    csv_path = os.path.join(output_dir, f'popularity_stratification_{dataset_name}.csv')
    df_res.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")
    
    # 4. Generate Publication-Quality Bar Chart
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: NDCG@20 by Strata
    plt.subplot(1, 2, 1)
    strata = ['Overall', 'Head', 'Mid', 'Tail']
    base_ndcg = [res_base['NDCG@20 (Overall)'], res_base['NDCG@20 (Head)'], res_base['NDCG@20 (Mid)'], res_base['NDCG@20 (Tail)']]
    fs_ndcg = [res_fs['NDCG@20 (Overall)'], res_fs['NDCG@20 (Head)'], res_fs['NDCG@20 (Mid)'], res_fs['NDCG@20 (Tail)']]
    
    x = np.arange(len(strata))
    width = 0.35
    
    plt.bar(x - width/2, base_ndcg, width, label='Vanilla SGCL', color='#7f8c8d')
    plt.bar(x + width/2, fs_ndcg, width, label='FS-SGCL (Ours)', color='#2980b9')
    
    for i in range(len(strata)):
        gain = ((fs_ndcg[i] - base_ndcg[i]) / base_ndcg[i]) * 100
        plt.text(x[i] + width/2, fs_ndcg[i] + 0.001, f"+{gain:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1b4f72')
        
    plt.ylabel('NDCG@20', fontsize=11)
    plt.title(f'NDCG@20 by Item Popularity Strata\n({dataset_name})', fontsize=12, fontweight='bold')
    plt.xticks(x, strata, fontsize=10)
    plt.legend(frameon=True)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Subplot 2: Recommendation Exposure Distribution
    plt.subplot(1, 2, 2)
    categories = ['Head Items', 'Mid Items', 'Tail Items']
    base_shares = [res_base['Rec Share Head (%)'], res_base['Rec Share Mid (%)'], res_base['Rec Share Tail (%)']]
    fs_shares = [res_fs['Rec Share Head (%)'], res_fs['Rec Share Mid (%)'], res_fs['Rec Share Tail (%)']]
    
    x2 = np.arange(len(categories))
    plt.bar(x2 - width/2, base_shares, width, label='Vanilla SGCL', color='#95a5a6')
    plt.bar(x2 + width/2, fs_shares, width, label='FS-SGCL (Ours)', color='#27ae60')
    
    for i in range(len(categories)):
        plt.text(x2[i] - width/2, base_shares[i] + 0.5, f"{base_shares[i]:.1f}%", ha='center', va='bottom', fontsize=9)
        plt.text(x2[i] + width/2, fs_shares[i] + 0.5, f"{fs_shares[i]:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    plt.ylabel('Recommendation Share (%) in Top-20', fontsize=11)
    plt.title(f'Popularity Distribution of Top-20 Recommendations\n({dataset_name})', fontsize=12, fontweight='bold')
    plt.xticks(x2, categories, fontsize=10)
    plt.legend(frameon=True)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'ndcg_by_popularity_{dataset_name}.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved plot: {plot_path}")
    
    return df_res

if __name__ == '__main__':
    # Toys & Games
    toys_base = 'saved/SGCL-Oct-25-2026_22-26-11.pth'
    toys_fs = 'saved/SGCL-Nov-01-2026_12-23-24.pth'
    if os.path.exists(toys_base) and os.path.exists(toys_fs):
        evaluate_popularity_strata('amazon-toys-games', toys_base, toys_fs)
        
    # Beauty
    beauty_base = 'saved/SGCL-Oct-25-2026_21-32-31.pth'
    beauty_fs = 'saved/SGCL-Nov-01-2026_13-27-58.pth'
    if os.path.exists(beauty_base) and os.path.exists(beauty_fs):
        evaluate_popularity_strata('amazon-beauty', beauty_base, beauty_fs)
