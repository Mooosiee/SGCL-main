import torch
import torch.nn.functional as F

print("--- Testing Pop-SGCL Loss and Gradient Flow ---")

batch_size = 1024
dim = 64
n_users = 22364
n_items = 12102
temperature = 0.2
pop_beta = 0.5
pop_alpha = 0.2

# Synthetic batch
user = torch.randint(1, n_users, (batch_size,))
pos_item = torch.randint(1, n_items, (batch_size,))

# Insert intentional duplicates to test masking
user[10] = user[0]
pos_item[10] = pos_item[0]
pos_item[20] = pos_item[5]

# Synthetic degrees
user_deg = torch.randint(1, 100, (n_users,)).float()
item_deg = torch.randint(1, 300, (n_items,)).float()

# Ego embeddings with requires_grad
u_emb = torch.randn(batch_size, dim, requires_grad=True)
i_emb = torch.randn(batch_size, dim, requires_grad=True)

u_norm = F.normalize(u_emb, dim=-1)
i_norm = F.normalize(i_emb, dim=-1)

pos_score = (u_norm * i_norm).sum(dim=-1)
pos_score = torch.exp(pos_score / temperature)

ttl_u_sim = torch.matmul(u_norm, u_norm.transpose(0, 1)) / temperature
ttl_i_sim = torch.matmul(i_norm, i_norm.transpose(0, 1)) / temperature

# Popularity correction
if pop_alpha > 0:
    u_penalty = pop_alpha * torch.log(user_deg[user] + 1.0)
    ttl_u_sim = ttl_u_sim - u_penalty.unsqueeze(0)

if pop_beta > 0:
    i_penalty = pop_beta * torch.log(item_deg[pos_item] + 1.0)
    ttl_i_sim = ttl_i_sim - i_penalty.unsqueeze(0)

# Masking
u_mask = (user.unsqueeze(1) == user.unsqueeze(0))
i_mask = (pos_item.unsqueeze(1) == pos_item.unsqueeze(0))
ttl_u_sim = ttl_u_sim.masked_fill(u_mask, -1e9)
ttl_i_sim = ttl_i_sim.masked_fill(i_mask, -1e9)

ttl_u_score = torch.exp(ttl_u_sim).sum(dim=1)
ttl_i_score = torch.exp(ttl_i_sim).sum(dim=1)

ttl_score = ttl_u_score + ttl_i_score
loss = -torch.log(pos_score / (ttl_score + 1e-8)).sum()

print(f"Loss value: {loss.item():.4f}")
loss.backward()
print(f"u_emb grad norm: {u_emb.grad.norm().item():.4f}")
print(f"i_emb grad norm: {i_emb.grad.norm().item():.4f}")
print("Gradient flow verified successfully!")
