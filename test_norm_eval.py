import torch
import torch.nn.functional as F
from recbole_gnn.config import Config
from recbole_gnn.utils import create_dataset, data_preparation, get_model, get_trainer

print("--- Testing Inference Normalization on Pop-FN-SGCL Checkpoint ---")

config = Config(model='SGCL', dataset='amazon-beauty', config_file_list=['config.yaml'])
dataset = create_dataset(config)
train_data, valid_data, test_data = data_preparation(config, dataset)

ckpt_path = 'saved/SGCL-Oct-31-2026_19-26-43.pth'
checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

model = get_model('SGCL')(config, train_data.dataset)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

trainer = get_trainer(config['MODEL_TYPE'], 'SGCL')(config, model)
trainer.eval_collector.data_collect(train_data)

# 1. Standard evaluation (unnormalized dot product)
print("\n1. Standard Evaluation (Unnormalized Dot Product):")
res_unnorm = trainer.evaluate(test_data, load_best_model=False, show_progress=False)
print("Unnormalized Test:", res_unnorm)

# 2. Monkey-patch full_sort_predict to use normalized embeddings
orig_full_sort = model.full_sort_predict

def normalized_full_sort_predict(interaction):
    user = interaction[model.USER_ID]
    if model.restore_user_e is None or model.restore_item_e is None:
        u_e, i_e = model.forward()
        model.restore_user_e = F.normalize(u_e, dim=-1)
        model.restore_item_e = F.normalize(i_e, dim=-1)
    u_embeddings = model.restore_user_e[user]
    scores = torch.matmul(u_embeddings, model.restore_item_e.transpose(0, 1))
    return scores.view(-1)

model.full_sort_predict = normalized_full_sort_predict
model.restore_user_e = None
model.restore_item_e = None

print("\n2. Normalized Evaluation (Cosine Similarity):")
res_norm = trainer.evaluate(test_data, load_best_model=False, show_progress=False)
print("Normalized Test:", res_norm)
