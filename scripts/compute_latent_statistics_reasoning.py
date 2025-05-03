import sys

sys.path.append(".")
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from dictionary_learning.dictionary import BatchTopKCrossCoder as CrossCoder
from datasets import load_dataset
from loguru import logger
import torch as th
from pathlib import Path
import os
import json
from tools.latent_analysis import latent_statistics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--crosscoder-path",
        type=str,
        default="/disk/u/troitskiid/data/checkpoints/DeepScaleR-1.5B-crosscoder-L15-k100-lr1e-04-local-shuffling-CCLoss/ae.pt",
    )
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--reasoning-model", type=str, default="agentica-org/DeepScaleR-1.5B-Preview")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--subset-size", type=int, default=100000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default="stats",
    )
    parser.add_argument(
        "--base-dataset",
        type=str,
        default="science-of-finetuning/fineweb-1m-sample",
        help="Dataset to compute statistics for base model",
    )
    parser.add_argument(
        "--reasoning-dataset",
        type=str,
        default="koyena/OpenR1-Math-220k-formatted",
        help="Dataset to compute statistics for reasoning model",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=5000,
        help="Max length of the input",
    )
    parser.add_argument(
        "--dataset-base-split",
        type=str,
        default="validation",
        help="Split of the base dataset to use",
    )
    parser.add_argument(
        "--dataset-reasoning-split",
        type=str,
        default="test",
        help="Split of the reasoning dataset to use",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not th.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"
    device = th.device(args.device)

    # Load models
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=th.bfloat16,
        device_map=device,
        attn_implementation="eager",
    )
    reasoning_model = AutoModelForCausalLM.from_pretrained(
        args.reasoning_model,
        torch_dtype=th.bfloat16,
        device_map=device,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.reasoning_model)

    # Load crosscoder
    cc = CrossCoder.from_pretrained(args.crosscoder_path, from_hub=False)
    text_column_reasoning = "message_in_chat_template"
    cc = cc.to(device)

    # Load datasets
    logger.info(f"Loading base dataset from {args.base_dataset}")
    validation_set_base = load_dataset(args.base_dataset, split=args.dataset_base_split)
    validation_set_base = validation_set_base.select(
        range(min(args.subset_size, len(validation_set_base)))
    )

    logger.info(f"Loading reasoning dataset from {args.reasoning_dataset}")
    validation_set_reasoning = load_dataset(args.reasoning_dataset, split=args.dataset_reasoning_split)
    validation_set_reasoning = validation_set_reasoning.select(
        range(min(args.subset_size, len(validation_set_reasoning)))
    )

    logger.info("Computing statistics for base dataset...")
    stats_fineweb = latent_statistics(
        validation_set_base.select_columns("text"),
        tokenizer,
        base_model,
        reasoning_model,
        cc,
        args.layer,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    logger.info("Computing statistics for reasoning dataset...")
    stats_reasoning = latent_statistics(
        validation_set_reasoning.select_columns(text_column_reasoning),
        tokenizer,
        base_model,
        reasoning_model,
        cc,
        args.layer,
        batch_size=args.batch_size,
        text_column=text_column_reasoning,
        max_length=args.max_length,
    )

    # Save results
    results_dir = args.results_dir / "_".join(args.crosscoder_path.split("/")[-2:])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save args
    with open(results_dir / "args.json", "w") as f:
        json.dump(
            {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f
        )

    th.save(stats_fineweb, results_dir / "fineweb.pt")
    th.save(stats_reasoning, results_dir / "reasoning.pt")


if __name__ == "__main__":
    main()
