import torch
from recbole_gnn.config import Config
from recbole_gnn.utils import create_dataset, data_preparation, get_model, get_trainer

print("--- Evaluating Toys-and-Games Pop-FN-SGCL Checkpoint (Epoch 26) ---")

config_dict = {
    'mask_mode': 'all',
    'pop_beta': 0.5,
    'pop_alpha': 0.0,
}
config = Config(model='SGCL', dataset='amazon-toys-games', config_file_list=['config.yaml'], config_dict=config_dict)
dataset = create_dataset(config)
train_data, valid_data, test_data = data_preparation(config, dataset)

ckpt_path = 'saved/SGCL-Oct-31-2026_22-25-52.pth'
checkpoint = torch.load(ckpt_path, map_location='cpu')

model = get_model('SGCL')(config, train_data.dataset)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

trainer = get_trainer(config['MODEL_TYPE'], 'SGCL')(config, model)

print("\nEvaluating on Test Set:")
test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=False)
print("\nFinal Test Result for Pop-FN-SGCL on Amazon-Toys-Games:")
for k, v in test_result.items():
    print(f"  {k}: {v:.4f}")
