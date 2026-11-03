import os
import torch
import numpy as np
from collections import Counter
from recbole_gnn.config import Config
from recbole_gnn.utils import create_dataset, data_preparation

def profile_dataset(dataset_name):
    print(f"\n{'='*70}")
    print(f" PROFILING BATCH COLLISIONS & FALSE NEGATIVES: {dataset_name}")
    print(f"{'='*70}")

    config = Config(
        model='SGCL',
        dataset=dataset_name,
        config_file_list=['config.yaml'],
        config_dict={'show_progress': False, 'save_dataloaders': False}
    )
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    total_batches = 0
    total_samples = 0
    total_comparisons = 0
    
    # User collision trackers
    u_dup_pairs_total = 0
    u_collision_batches = 0
    u_unique_per_batch = []
    u_max_dup_in_batch = 0

    # Item collision trackers
    i_dup_pairs_total = 0
    i_collision_batches = 0
    i_unique_per_batch = []
    i_max_dup_in_batch = 0

    # User & Item degree tracking for collided entities
    u_collision_counter = Counter()
    i_collision_counter = Counter()
    
    # Store per-batch stats
    u_dups_per_batch_list = []
    i_dups_per_batch_list = []

    temperature = config['temperature'] if 'temperature' in config else 0.2
    max_penalty_per_collision = np.exp(1.0 / temperature) # exp(5) ~ 148.41

    for batch_idx, interaction in enumerate(train_data):
        users = interaction[config['USER_ID_FIELD']]
        items = interaction[config['ITEM_ID_FIELD']]
        B = len(users)
        total_batches += 1
        total_samples += B
        
        # Total off-diagonal pairs in (B, B) matrix: B * (B - 1)
        off_diag_pairs = B * (B - 1)
        total_comparisons += off_diag_pairs

        # User collisions: u_j == u_k for j != k
        u_match = (users.unsqueeze(1) == users.unsqueeze(0)).fill_diagonal_(False)
        u_dups = u_match.sum().item() # counts directed pairs (j, k)
        u_dup_pairs_total += u_dups
        u_dups_per_batch_list.append(u_dups)
        if u_dups > 0:
            u_collision_batches += 1
            if u_dups > u_max_dup_in_batch:
                u_max_dup_in_batch = u_dups
            # Identify which users collided
            collided_u = users[u_match.any(dim=1)].tolist()
            u_collision_counter.update(collided_u)

        u_unique = len(torch.unique(users))
        u_unique_per_batch.append(u_unique)

        # Item collisions: i_j == i_k for j != k
        i_match = (items.unsqueeze(1) == items.unsqueeze(0)).fill_diagonal_(False)
        i_dups = i_match.sum().item()
        i_dup_pairs_total += i_dups
        i_dups_per_batch_list.append(i_dups)
        if i_dups > 0:
            i_collision_batches += 1
            if i_dups > i_max_dup_in_batch:
                i_max_dup_in_batch = i_dups
            # Identify which items collided
            collided_i = items[i_match.any(dim=1)].tolist()
            i_collision_counter.update(collided_i)

        i_unique = len(torch.unique(items))
        i_unique_per_batch.append(i_unique)

    # Summary calculations
    avg_batch_size = total_samples / total_batches
    avg_u_dups = u_dup_pairs_total / total_batches
    avg_i_dups = i_dup_pairs_total / total_batches

    print(f"\n[1] Overall Dataset & Training Statistics:")
    print(f"  - Total Interactions in Train: {total_samples:,}")
    print(f"  - Total Batches per Epoch:     {total_batches}")
    print(f"  - Nominal Batch Size:          {config['train_batch_size']}")
    print(f"  - Average Actual Batch Size:   {avg_batch_size:.1f}")
    print(f"  - Total Pairwise Comparisons:  {total_comparisons:,}")

    print(f"\n[2] User-Side Collisions (False Negatives in ttl_u_score):")
    print(f"  - Batches with User Collisions:       {u_collision_batches}/{total_batches} ({u_collision_batches/total_batches*100:.1f}%)")
    print(f"  - Avg Duplicate User Pairs / Batch:   {avg_u_dups:.2f} pairs")
    print(f"  - Max Duplicate User Pairs in a Batch:{u_max_dup_in_batch} pairs")
    print(f"  - Avg Unique Users per 1024 Batch:    {np.mean(u_unique_per_batch):.1f} / {avg_batch_size:.0f} ({np.mean(u_unique_per_batch)/avg_batch_size*100:.1f}%)")
    print(f"  - Total False-Negative User Pairs:    {u_dup_pairs_total:,} per epoch")

    print(f"\n[3] Item-Side Collisions (False Negatives in ttl_i_score):")
    print(f"  - Batches with Item Collisions:       {i_collision_batches}/{total_batches} ({i_collision_batches/total_batches*100:.1f}%)")
    print(f"  - Avg Duplicate Item Pairs / Batch:   {avg_i_dups:.2f} pairs")
    print(f"  - Max Duplicate Item Pairs in a Batch:{i_max_dup_in_batch} pairs")
    print(f"  - Avg Unique Items per 1024 Batch:    {np.mean(i_unique_per_batch):.1f} / {avg_batch_size:.0f} ({np.mean(i_unique_per_batch)/avg_batch_size*100:.1f}%)")
    print(f"  - Total False-Negative Item Pairs:    {i_dup_pairs_total:,} per epoch")

    print(f"\n[4] Gradient Corruption & Popularity Concentration:")
    print(f"  - Contrastive Temperature (tau):      {temperature}")
    print(f"  - Maximal False Penalty per Collision:exp(1/tau) = exp(5) = {max_penalty_per_collision:.2f}")
    print(f"  - Total Artificial Repulsion Mass:    {(u_dup_pairs_total + i_dup_pairs_total) * max_penalty_per_collision:,.0f} per epoch")
    print(f"  - Unique Colliding Items:             {len(i_collision_counter)} out of {dataset.item_num} total items")
    print(f"  - Top-5 Most Pushed-Away Items IDs:   {i_collision_counter.most_common(5)}")

    return {
        'dataset': dataset_name,
        'total_batches': total_batches,
        'avg_batch_size': avg_batch_size,
        'u_dup_avg': avg_u_dups,
        'u_collision_batch_pct': u_collision_batches / total_batches * 100,
        'i_dup_avg': avg_i_dups,
        'i_collision_batch_pct': i_collision_batches / total_batches * 100,
        'u_unique_avg': float(np.mean(u_unique_per_batch)),
        'i_unique_avg': float(np.mean(i_unique_per_batch)),
        'total_fn_pairs': u_dup_pairs_total + i_dup_pairs_total,
    }

if __name__ == '__main__':
    results = []
    for d in ['amazon-beauty', 'amazon-toys-games']:
        res = profile_dataset(d)
        results.append(res)
    
    print("\n" + "="*70)
    print(" SUMMARY TABLE FOR REPORT / PAPER")
    print("="*70)
    print(f"{'Dataset':<20} | {'User Collisions/batch':<22} | {'Item Collisions/batch':<22} | {'Item Collision Batches %'}")
    print("-" * 80)
    for r in results:
        print(f"{r['dataset']:<20} | {r['u_dup_avg']:<22.2f} | {r['i_dup_avg']:<22.2f} | {r['i_collision_batch_pct']:.1f}%")
