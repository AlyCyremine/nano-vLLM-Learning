import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    if param.shape != loaded_weight.shape:
        raise ValueError(f"Weight shape mismatch: expected {tuple(param.shape)}, got {tuple(loaded_weight.shape)}")
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    map_weight_name = getattr(model, "map_weight_name", lambda name: name)
    files = sorted(glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No safetensors weights found in {path}")
    loaded = set()
    shards = {}
    skipped = 0
    for file in files:
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                original_name = weight_name
                weight_name = map_weight_name(weight_name)
                if weight_name is None:
                    skipped += 1
                    continue
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(original_name), shard_id)
                        loaded.add(param_name)
                        shards.setdefault(param_name, set()).add(shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(original_name))
                    loaded.add(weight_name)
    tied = getattr(model, "tied_weights_mapping", {})
    missing = [name for name, _ in model.named_parameters() if name not in loaded and tied.get(name) not in loaded]
    if missing:
        raise ValueError(f"Missing model weights: {missing}")
    for name, actual_shards in shards.items():
        expected = {shard for source, (target, shard) in packed_modules_mapping.items() if target in name}
        if actual_shards != expected:
            raise ValueError(f"Incomplete packed weight {name}: expected {expected}, got {actual_shards}")
    return {"loaded_parameters": len(loaded), "skipped_weights": skipped}
