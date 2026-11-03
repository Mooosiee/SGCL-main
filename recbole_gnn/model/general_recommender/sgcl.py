# -*- coding: utf-8 -*-
r"""
SGCL
################################################
"""
import torch
import torch.nn.functional as F
from recbole_gnn.model.general_recommender import LightGCN
from recbole.utils import InputType

class SGCL(LightGCN):
    def __init__(self, config, dataset):
        super(SGCL, self).__init__(config, dataset)
        self.temperature = config['temperature']
        # mask_mode: 'all' (mask self + batch duplicates), 'diagonal' (self only), 'none' (original SGCL)
        self.mask_mode = config['mask_mode'] if 'mask_mode' in config else 'all'
        # Popularity calibration coefficients (logQ correction on negative denominator)
        self.pop_beta = float(config['pop_beta']) if ('pop_beta' in config and config['pop_beta'] is not None) else 0.0
        self.pop_alpha = float(config['pop_alpha']) if ('pop_alpha' in config and config['pop_alpha'] is not None) else 0.0
        # FS-SGCL enhancements: cross-domain user-item negatives and layer weighting
        self.cross_weight = float(config['cross_weight']) if ('cross_weight' in config and config['cross_weight'] is not None) else 0.0
        self.layer_weight_mode = config['layer_weight_mode'] if ('layer_weight_mode' in config and config['layer_weight_mode'] is not None) else 'mean'

        if self.layer_weight_mode == 'learnable':
            init_w = [1.5, 1.0, 0.5, 0.2][:self.n_layers + 1]
            while len(init_w) < self.n_layers + 1:
                init_w.append(0.1)
            self.layer_weights = torch.nn.Parameter(torch.tensor(init_w, dtype=torch.float))

        # Precompute interaction degree distributions for users and items
        if hasattr(dataset, 'inter_feat') and dataset.inter_feat is not None:
            u_inter = dataset.inter_feat[dataset.uid_field]
            i_inter = dataset.inter_feat[dataset.iid_field]
            if not isinstance(u_inter, torch.Tensor):
                u_inter = torch.tensor(u_inter, dtype=torch.long)
            if not isinstance(i_inter, torch.Tensor):
                i_inter = torch.tensor(i_inter, dtype=torch.long)
            user_deg = torch.bincount(u_inter, minlength=self.n_users).float()
            item_deg = torch.bincount(i_inter, minlength=self.n_items).float()
        else:
            user_deg = torch.ones(self.n_users, dtype=torch.float)
            item_deg = torch.ones(self.n_items, dtype=torch.float)

        self.register_buffer('user_deg', user_deg)
        self.register_buffer('item_deg', item_deg)

    def load_state_dict(self, state_dict, strict=True):
        # Backward compatibility for legacy checkpoints
        if 'user_deg' not in state_dict and hasattr(self, 'user_deg'):
            state_dict['user_deg'] = self.user_deg
        if 'item_deg' not in state_dict and hasattr(self, 'item_deg'):
            state_dict['item_deg'] = self.item_deg
        if 'layer_weights' not in state_dict and hasattr(self, 'layer_weights'):
            state_dict['layer_weights'] = self.layer_weights
        return super(SGCL, self).load_state_dict(state_dict, strict=strict)
        
    def forward(self):
        all_embs = self.get_ego_embeddings()
        embeddings_list = [all_embs]

        for layer_idx in range(self.n_layers):
            all_embs = self.gcn_conv(all_embs, self.edge_index, self.edge_weight)
            embeddings_list.append(all_embs)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        
        if self.layer_weight_mode == 'learnable':
            w = F.softmax(self.layer_weights, dim=0)
            lightgcn_all_embeddings = (lightgcn_all_embeddings * w.view(1, -1, 1)).sum(dim=1)
        elif self.layer_weight_mode == 'decay':
            decay_w = torch.tensor([0.4, 0.3, 0.2, 0.1][:self.n_layers + 1], device=lightgcn_all_embeddings.device)
            decay_w = decay_w / decay_w.sum()
            lightgcn_all_embeddings = (lightgcn_all_embeddings * decay_w.view(1, -1, 1)).sum(dim=1)
        else:
            lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings
    
    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        
        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[pos_item]

        u_embeddings, i_embeddings = F.normalize(u_embeddings, dim=-1), F.normalize(i_embeddings, dim=-1)
        pos_score = (u_embeddings * i_embeddings).sum(dim=-1)
        pos_score = torch.exp(pos_score / self.temperature)

        ttl_u_sim = torch.matmul(u_embeddings, u_embeddings.transpose(0, 1)) / self.temperature
        ttl_i_sim = torch.matmul(i_embeddings, i_embeddings.transpose(0, 1)) / self.temperature

        # Popularity / Degree Calibration (LogQ Correction on Negative Denominator)
        if self.pop_alpha > 0:
            u_penalty = self.pop_alpha * torch.log(self.user_deg[user] + 1.0)
            ttl_u_sim = ttl_u_sim - u_penalty.unsqueeze(0)

        if self.pop_beta > 0:
            i_penalty = self.pop_beta * torch.log(self.item_deg[pos_item] + 1.0)
            ttl_i_sim = ttl_i_sim - i_penalty.unsqueeze(0)

        if self.mask_mode == 'all':
            # Mask out identical IDs (naturally includes diagonal where user[j] == user[j])
            u_mask = (user.unsqueeze(1) == user.unsqueeze(0))
            i_mask = (pos_item.unsqueeze(1) == pos_item.unsqueeze(0))
            ttl_u_sim = ttl_u_sim.masked_fill(u_mask, -1e9)
            ttl_i_sim = ttl_i_sim.masked_fill(i_mask, -1e9)
        elif self.mask_mode == 'diagonal':
            # Mask only diagonal self-comparisons (j == k)
            diag_mask = torch.eye(len(user), dtype=torch.bool, device=user.device)
            ttl_u_sim = ttl_u_sim.masked_fill(diag_mask, -1e9)
            ttl_i_sim = ttl_i_sim.masked_fill(diag_mask, -1e9)
        elif self.mask_mode == 'none':
            pass

        ttl_u_score = torch.exp(ttl_u_sim).sum(dim=1)
        ttl_i_score = torch.exp(ttl_i_sim).sum(dim=1)

        ttl_score = ttl_u_score + ttl_i_score

        # FS-SGCL: Cross-Domain User-Item Negative Contrast
        if self.cross_weight > 0:
            ttl_ui_sim = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1)) / self.temperature
            if self.pop_beta > 0:
                ttl_ui_sim = ttl_ui_sim - i_penalty.unsqueeze(0)
            if self.mask_mode == 'all':
                ui_mask = (user.unsqueeze(1) == user.unsqueeze(0)) | (pos_item.unsqueeze(1) == pos_item.unsqueeze(0))
                ttl_ui_sim = ttl_ui_sim.masked_fill(ui_mask, -1e9)
            elif self.mask_mode == 'diagonal':
                ui_diag = torch.eye(len(user), dtype=torch.bool, device=user.device)
                ttl_ui_sim = ttl_ui_sim.masked_fill(ui_diag, -1e9)
            ttl_ui_score = torch.exp(ttl_ui_sim).sum(dim=1)
            ttl_score = ttl_score + self.cross_weight * ttl_ui_score

        sup_cl_loss = -torch.log(pos_score / (ttl_score + 1e-8)).sum()

        return sup_cl_loss
    


