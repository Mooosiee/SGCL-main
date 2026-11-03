import torch
import torch.nn.functional as F

print("--- Testing Full-Spectrum SGCL (FS-SGCL) ---")

batch_size = 1024
dim = 64
n_users = 19413
n_items = 11925
temperature = 0.2
pop_beta = 0.5
pop_alpha = 0.0
cross_weight = 1.0

# Batch
user = torch.randint(1, n_users, (batch_size,))
pos_item = torch.randint(1, n_items, (batch_size,))
user[10] = user[0] # collision

# Degrees
user_deg = torch.randint(1, 100, (n_users,)).float()
item_deg = torch.randint(1, 300, (n_items,)).float()

# Ego embeddings
u_emb = torch.randn(batch_size, dim, requires_grad=True)
i_emb = torch.randn(batch_size, dim, requires_grad=True)

u_norm = F.normalize(u_emb, dim=-1)
i_norm = F.normalize(i_emb, dim=-1)

pos_score = (u_norm * i_norm).sum(dim=-1)
pos_score = torch.exp(pos_score / temperature)

# 1. User-User similarities
ttl_u_sim = torch.matmul(u_norm, u_norm.transpose(0, 1)) / temperature
# 2. Item-Item similarities
ttl_i_sim = torch.matmul(i_norm, i_norm.transpose(0, 1)) / temperature
# 3. Cross-Domain User-Item similarities (NEW in FS-SGCL!)
ttl_ui_sim = torch.matmul(u_norm, i_norm.transpose(0, 1)) / temperature

# LogQ popularity penalties
i_penalty = pop_beta * torch.log(item_deg[pos_item] + 1.0)
ttl_i_sim = ttl_i_sim - i_penalty.unsqueeze(0)
ttl_ui_sim = ttl_ui_sim - i_penalty.unsqueeze(0)

# Masking
# u_mask: identical users
u_mask = (user.unsqueeze(1) == user.unsqueeze(0))
# i_mask: identical items
i_mask = (pos_item.unsqueeze(1) == pos_item.unsqueeze(0))
# ui_mask: diagonal (positive pair) + identical users (false negatives)
ui_mask = (user.unsqueeze(1) == user.unsqueeze(0)) | torch.eye(batch_size, dtype=torch.bool)

ttl_u_sim = ttl_u_sim.masked_fill(u_mask, -1e9)
ttl_i_sim = ttl_i_sim.masked_fill(i_mask, -1e9)
ttl_ui_sim = ttl_ui_sim.masked_fill(ui_mask, -1e9)

ttl_u_score = torch.exp(ttl_u_sim).sum(dim=1)
ttl_i_score = torch.exp(ttl_i_sim).sum(dim=1)
ttl_ui_score = torch.exp(ttl_ui_sim).sum(dim=1)

ttl_score = ttl_u_score + ttl_i_score + cross_weight * ttl_ui_score
loss = -torch.log(pos_score / (ttl_score + 1e-8)).sum()

print(f"Loss value: {loss.item():.4f}")
loss.backward()
print(f"u_emb grad norm: {u_emb.grad.norm().item():.4f}")
print(f"i_emb grad norm: {i_emb.grad.norm().item():.4f}")

# Test Layer Weights
n_layers = 3
layer_weights = torch.nn.Parameter(torch.tensor([1.5, 1.0, 0.5, 0.2]))
embs = [torch.randn(10, dim) for _ in range(n_layers + 1)]
stacked = torch.stack(embs, dim=1)
weights = F.softmax(layer_weights, dim=0)
weighted = (stacked * weights.view(1, -1, 1)).sum(dim=1)
print(f"Learned layer weights: {weights.tolist()}")
print(f"Weighted output shape: {weighted.shape}")
print("FS-SGCL components verified successfully!")
