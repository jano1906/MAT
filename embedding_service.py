import sys
import os
import torch
import numpy as np
from tqdm import tqdm
from typing import Optional, List, Literal
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from transformer import make_model, GraphTransformer
from featurization.data_utils import load_data_from_smiles, construct_loader

def checkpoint_path(model_name: str):
    return os.path.join(os.path.dirname(__file__), f"{model_name}.ckpt")

CHECKPOINT_DOWNLOAD_LINKS = {
    "MAT": "https://drive.google.com/open?id=11-TZj8tlnD7ykQGliO9bCrySJNBnYD2k",
}

class State:
    model: Optional[GraphTransformer] = None

    model_name: Optional[str] = None
    device: Optional[str] = None
    batch_size: Optional[int] = None

    initialized: bool = False

def setup(model_name: str, device: Literal["cpu", "cuda"], batch_size: int) -> None:
    if not os.path.isfile(checkpoint_path(model_name)):
        raise RuntimeError(f"Download checkpoint '{CHECKPOINT_DOWNLOAD_LINKS[model_name]}' and save it as '{checkpoint_path(model_name)}'.")
    
    model_params = {
    'd_atom': 28,
    'd_model': 1024,
    'N': 8,
    'h': 16,
    'N_dense': 1,
    'lambda_attention': 0.33, 
    'lambda_distance': 0.33,
    'leaky_relu_slope': 0.1, 
    'dense_output_nonlinearity': 'relu', 
    'distance_matrix_kernel': 'exp', 
    'dropout': 0.0,
    'aggregation_type': 'mean',}
    model = make_model(**model_params)
    model.generator.proj = torch.nn.Identity()
    pretrained_state_dict = torch.load(checkpoint_path(model_name))
    model_state_dict = model.state_dict()
    for name, param in pretrained_state_dict.items():
        if 'generator' in name:
            continue
        if isinstance(param, torch.nn.Parameter):
            param = param.data
        model_state_dict[name].copy_(param)
    model = model.to(device)
    model.eval()
    
    State.model = model
    State.model_name = model_name
    State.device = device
    State.batch_size = batch_size
    State.initialized = True

def encode(smiles: List[str]) -> np.ndarray:
    if not State.initialized:
        raise RuntimeError("Service is not setup, call 'setup' before 'encode'.")
    
    X, y = load_data_from_smiles(smiles, [0.]*len(smiles), add_dummy_node=True, one_hot_formal_charge=True)
    data_loader = construct_loader(X, y, State.batch_size)
    outputs = []
    with torch.no_grad():
        for batch in tqdm(data_loader, f"Encoding with {State.model_name}"):
            adjacency_matrix, node_features, distance_matrix, y = batch
            
            adjacency_matrix = adjacency_matrix.to(State.device)
            node_features = node_features.to(State.device)
            distance_matrix = distance_matrix.to(State.device)
            batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
            
            output = State.model(node_features, batch_mask, adjacency_matrix, distance_matrix, None)
            outputs.append(output)
            
    outputs = torch.concat(outputs)
    outputs = outputs.cpu().numpy()
    return outputs
