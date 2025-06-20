from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 10,
        "lr":  1e-4,
        "seq_len": 500,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "de",
        "dataset_split": "de-en",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None, 
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "run/tmodel"
    }

def get_weight_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)