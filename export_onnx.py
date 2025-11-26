

import argparse
import torch

from model import TransformerModel    
from config import get_config          


def parse_args():
    parser = argparse.ArgumentParser(description="Export Transformer seq2seq model to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained Transformer checkpoint (.pt/.pth)",
    )
    parser.add_argument(
        "--onnx-out",
        type=str,
        default="transformer_model.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--src-max-len",
        type=int,
        default=64,
        help="Max source sequence length for dummy input",
    )
    parser.add_argument(
        "--tgt-max-len",
        type=int,
        default=64,
        help="Max target sequence length for dummy input",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ----- Load config & model -----
    config = get_config()

    src_vocab_size = config.get("src_vocab_size")
    tgt_vocab_size = config.get("tgt_vocab_size")

    if src_vocab_size is None or tgt_vocab_size is None:
        raise ValueError(
            "Config must contain 'src_vocab_size' and 'tgt_vocab_size' for dummy input generation."
        )

    model = TransformerModel(config)
    state = torch.load(args.checkpoint, map_location="cpu")
    # Adjust if checkpoint dict structure differs
    model.load_state_dict(state)
    model.eval()

    # ----- Dummy source & target tokens -----
    src_tokens = torch.randint(
        low=0,
        high=src_vocab_size,
        size=(1, args.src_max_len),
        dtype=torch.long,
    )
    tgt_tokens = torch.randint(
        low=0,
        high=tgt_vocab_size,
        size=(1, args.tgt_max_len),
        dtype=torch.long,
    )

    # If your model forward signature is different, adjust the call below.
    # Common pattern: model(src_tokens, tgt_tokens)
    print(f"Exporting Transformer seq2seq model to ONNX: {args.onnx_out}")
    torch.onnx.export(
        model,
        (src_tokens, tgt_tokens),
        args.onnx_out,
        input_names=["src_tokens", "tgt_tokens"],
        output_names=["logits"],
        opset_version=args.opset,
        dynamic_axes={
            "src_tokens": {0: "batch_size", 1: "src_sequence_length"},
            "tgt_tokens": {0: "batch_size", 1: "tgt_sequence_length"},
            "logits": {0: "batch_size", 1: "tgt_sequence_length"},
        },
    )
    print("Done.")


if __name__ == "__main__":
    main()
