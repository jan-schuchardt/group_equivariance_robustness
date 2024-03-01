from src.models.utils import load_feature_extractor
from src.datasets.data_provider import get_dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import torch
import yaml

with open("../src/seml/configs/eval.yaml", "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

trainset, valset, testset = get_dataset("QM9", "U0", seed=1)
model = load_feature_extractor(
    "dimenet", config['fixed']['encoder_params'], 
    pretrained='/nfs/homedirs/wollschl/staff/uncertainty-molecules/models/dimenet_pretrained_U0')


model.to_encoder(output_dim=128)
device = torch.device("cuda")
model = model.to(device)
new_inputs = []
targets = []
with torch.no_grad():
    for batch in tqdm(DataLoader(trainset, batch_size=256)):
        batch = batch.to(device)
        x, y = (batch.z, batch.pos, batch.batch), batch.y
        new_inputs.append(model(*x).cpu())
        targets.append(y.reshape(-1))
new_inputs = torch.cat(new_inputs)
targets = torch.cat(targets)

dataset = {
    'inputs': new_inputs,
    'targets': targets
}
torch.save(dataset, "../data/dimenet_qm9_U0_dim_128")