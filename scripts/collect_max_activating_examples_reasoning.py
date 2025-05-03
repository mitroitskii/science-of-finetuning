"""
Script to collect max activating examples for a base and ft only latents of a CrossCoder.
"""
import sys
sys.path.append(".")
from huggingface_hub import hf_api
import wandb
from tqdm import tqdm
from nnterp import load_model
from nnterp.nnsight_utils import get_layer_output, get_layer
from datasets import load_dataset
from dictionary_learning import CrossCoder
from torch.utils.data import DataLoader
import torch as th
import json
import sqlite3
import gc
import argparse
import heapq
from multiprocessing import Process, Queue, cpu_count
from pathlib import Path
from tools.utils import load_crosscoder, load_latent_df



def max_act_exs_to_db(max_activating_examples, db_path: Path):
    """Convert max activating examples to a database."""

    print(f"Starting database conversion to {db_path}")

    print(f"Number of features to store: {len(max_activating_examples)}")

    if not db_path.exists():

        print(f"Creating new database at {db_path}")

        with sqlite3.connect(db_path) as conn:

            cursor = conn.cursor()

            print("Creating table schema...")

            cursor.execute(

                """CREATE TABLE IF NOT EXISTS data_table (

                    key INTEGER PRIMARY KEY,

                    examples TEXT

                )"""

            )

            print("Inserting examples into database...")

            for key, examples in max_activating_examples.items():

                print(f"Processing feature {key}: {len(examples)} examples")

                cursor.execute(

                    "INSERT INTO data_table (key, examples) VALUES (?, ?)",

                    (key, json.dumps(examples)),

                )

            print("Committing changes to database...")

            conn.commit()

            print("Database conversion completed successfully")


def sort_max_act_exs(max_activating_examples):
    """Sort max activating examples by activation value."""
    for feature_idx in max_activating_examples:
        max_activating_examples[feature_idx] = sorted(
            [(t[0], t[2], t[3]) for t in max_activating_examples[feature_idx]],
            key=lambda x: x[0],
            reverse=True,
        )
    return max_activating_examples


def cleanup_max_act_exs(max_activating, tokenizer, max_seq_len):
    """Clean up max activating examples by removing padding values and truncating sequences."""
    for feature_idx, examples in tqdm(max_activating.items(), desc="Cleaning up max activating examples"):
        for i, (ex_val, ex_str, ex_act) in enumerate(examples):
            # Remove padding values (-1) from the beginning of ex_act
            while ex_act and ex_act[0] == -1:
                ex_act.pop(0)
            examples[i] = (ex_val, ex_str, ex_act)
            # Truncate the end of ex_str to match the length of ex_act
            tokens = tokenizer.tokenize(ex_str, add_special_tokens=True)[
                :max_seq_len]
            tokens = [s.replace("▁", " ") for s in tokens]
            if len(tokens) != len(ex_act):
                print(
                    f"Warning: length of tokens {len(tokens)} does not match length of activation values {len(ex_act)} for example {i} of feature {feature_idx}"
                )
            examples[i] = (ex_val, tokens, ex_act)
    return max_activating


def merge_max_examples(*max_dicts):
    """Merge max activating examples from multiple dictionaries."""
    merged_dict = {}
    for d in max_dicts:
        for k, v in d.items():
            merged_dict[k] = merged_dict.get(k, []) + v
    # sort each list
    for k in merged_dict:
        merged_dict[k] = sorted(
            merged_dict[k], key=lambda x: x[0], reverse=True)
    return merged_dict


@th.no_grad()
def compute_max_activating_examples(
    dataset,
    feature_indices,
    crosscoder: CrossCoder,
    *,
    base_model,
    ft_model,
    save_path: Path,
    model_batch_size=64,
    crosscoder_batch_size=4096,
    n=50,
    layer=15,
    cc_device="cuda:7",
    workers=16,
    max_seq_len=5000,
    total_tokens=2_000_000,
    name="max_activating_examples",
    gc_collect_every=2,
    checkpoint_every=250,
) -> None:
    """Compute examples that maximally activate each feature in a CrossCoder model.

    Args:
        dataset: Dataset to search through for examples
        feature_indices: List of feature indices to find max activating examples for
        crosscoder: CrossCoder model to use for computing feature activations
        base_model: Base language model to get activations from
        ft_model: Fine-tuned model to get activations from
        save_path: Path to save results to
        model_batch_size: Batch size for running examples through base/ft models
        crosscoder_batch_size: Batch size for running activations through CrossCoder
        n: Number of max activating examples to find per feature
        layer: Which model layer to get activations from
        device: Device to run models on
        workers: Number of worker processes for data loading
        max_seq_len: Maximum sequence length to consider
        name: Name for saving results
        gc_collect_every: How often to run garbage collection
        checkpoint_every: How often to save checkpoints

    Returns:
        None. Results are saved to save_path/name.
    """
    if save_path is not None:
        save_path = save_path / name
        save_path.mkdir(parents=True, exist_ok=True)

    def dict_update_worker(
        queue, n, feature_indices, save_path, name, tokenizer, max_seq_len
    ):
        entry_id = 0  # entry id ensures we never compare the feat_act
        max_activating_examples = {k: [] for k in feature_indices}
        next_gb = gc_collect_every
        next_checkpoint = checkpoint_every
        num_samples = 0
        while True:
            item = queue.get()
            if item is None:  # Poison pill to stop the worker
                break
            max_activations, batch, feature_activations = item
            next_checkpoint -= 1
            next_gb -= 1
            num_samples += len(batch)
            if next_checkpoint <= 0 and save_path is not None:
                print(
                    f"Saving {num_samples} {name} examples as a checkpoint...")
                th.save(max_activating_examples,
                        save_path / f"{num_samples}.pt")
                next_checkpoint = checkpoint_every
            if next_gb <= 0:
                gc.collect()
                next_gb = gc_collect_every
            # Dictionary update logic (runs in separate process)
            for idx, feature_idx in enumerate(feature_indices):
                batch_values = max_activations[:, idx]
                entries = list(
                    zip(
                        batch_values.tolist(),
                        batch,
                        [feat_act[:, idx] for feat_act in feature_activations],
                    )
                )

                if len(max_activating_examples[feature_idx]) < n:
                    threshold = float("-inf")
                else:
                    threshold = max_activating_examples[feature_idx][0][0]

                potential_entries = [
                    entry for entry in entries if entry[0] > threshold]

                for entry in potential_entries:
                    if len(max_activating_examples[feature_idx]) < n:
                        heapq.heappush(
                            max_activating_examples[feature_idx],
                            (entry[0], entry_id, entry[1], entry[2].tolist()),
                        )
                        entry_id += 1
                    else:
                        heapq.heappushpop(
                            max_activating_examples[feature_idx],
                            (entry[0], entry_id, entry[1], entry[2].tolist()),
                        )
                        entry_id += 1

        max_activating_examples = sort_max_act_exs(max_activating_examples)
        max_activating_examples = cleanup_max_act_exs(
            max_activating_examples, tokenizer, max_seq_len
        )
        print(f"Saving {name} final examples...")
        th.save(max_activating_examples, save_path / f"{name}_final.pt")
        # convert to db
        print(f"(SKIP) Converting {name} final examples to db...")
        db_path = save_path / f"{name}_final.db"  # FIXME make sure this works
        max_act_exs_to_db(max_activating_examples, db_path)

    # Setup multiprocessing with bounded queue
    queue = Queue(maxsize=10)
    update_process = Process(
        target=dict_update_worker,
        args=(
            queue,
            n,
            feature_indices,
            save_path,
            name,
            ft_model.tokenizer,
            max_seq_len,
        ),
    )
    update_process.start()

    crosscoder.encoder.to(cc_device)
    dataloader = DataLoader(
        dataset, batch_size=model_batch_size, num_workers=workers)
    dec_weight = (
        crosscoder.decoder.weight.norm(dim=2).sum(
            dim=0, keepdim=True).to(cc_device)
    )
    num_tokens = 0

    # Main loop - now just collecting activations and queuing updates
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        # Check if the worker process is still alive
        if not update_process.is_alive():
            raise RuntimeError(
                "dict_update_worker process crashed unexpectedly.")

        bs = len(batch)
        tokens = ft_model.tokenizer(
            batch,
            max_length=max_seq_len,
            truncation=True,
            return_tensors="pt",
            padding=True,
        ).to(base_model.device)
        attention_mask = tokens["attention_mask"].to(cc_device)

        with base_model.trace(tokens):
            base_activations = get_layer_output(
                base_model, layer).to(cc_device).save()
            get_layer(base_model, layer).output.stop()
        with ft_model.trace(tokens):
            ft_activations = get_layer_output(
                ft_model, layer).to(cc_device).save()
            get_layer(ft_model, layer).output.stop()

        base_activations = base_activations.reshape(
            -1, base_activations.shape[-1])
        ft_activations = ft_activations.reshape(
            -1, ft_activations.shape[-1])
        merged_activations = th.stack(
            [base_activations, ft_activations], dim=1)

        feature_activations = []
        for act_batch in merged_activations.split(crosscoder_batch_size):
            feature_activations.append(
                (crosscoder.encoder(act_batch.float())
                 * dec_weight)[:, feature_indices].to(cc_device)
            )
        feature_activations = th.cat(feature_activations, dim=0)
        feature_activations = feature_activations.reshape(
            bs, -1, len(feature_indices))
        feature_activations = feature_activations.masked_fill(
            ~attention_mask.bool().unsqueeze(-1), -1
        )

        max_activations, _ = feature_activations.max(dim=1)
        num_tokens += attention_mask.sum().item()
        # Log metrics to wandb
        wandb.log(
            {
                "batch_idx": batch_idx,
                "mean_activation": max_activations.mean().item(),
                "max_activation": max_activations.max().item(),
                "min_activation": max_activations.min().item(),
                "queue_size": queue.qsize(),
                "num_tokens": num_tokens,
            }
        )

        # Queue the data for processing
        queue.put((max_activations.cpu(), batch, feature_activations.cpu()))

        if num_tokens >= total_tokens:
            print(
                f"\nReached token limit ({num_tokens} >= {total_tokens}). Stopping data collection for '{name}' after processing current batch.")
            break

    # Signal the worker to finish and get results
    queue.put(None)
    update_process.join()

    # Check if the worker process crashed
    if update_process.exitcode != 0:
        raise RuntimeError(
            "dict_update_worker process crashed with exit code: "
            f"{update_process.exitcode}"
        )

    return


# python scripts/collect_max_activating_examples.py connor --base-device "cuda:0" --ft-device "cuda:1" --lmsys-format base  --crosscoder-batch-size 512 --model-batch-size 16
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("crosscoder", type=str)
    parser.add_argument("--base-model", type=str,
                        default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--ft-model", type=str,
                        default="agentica-org/DeepScaleR-1.5B-Preview")
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--cc-device", type=str, default="cuda:7")
    parser.add_argument("--base-device", type=str, default="cuda:0")
    parser.add_argument("--ft-device", type=str, default="cuda:1")
    parser.add_argument("--total-tokens", type=int, default=2_000_000)
    parser.add_argument("--model-batch-size", type=int, default=64)
    parser.add_argument("--crosscoder-batch-size", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=5000)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--only-upload", action="store_true")
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("~/data/max_activating_examples"),
    )
    args = parser.parse_args()
    save_path = args.save_path / f"{args.crosscoder.split('/')[-2]}"

    if not args.only_upload:
        if args.workers is None:
            args.workers = cpu_count()

        # Initialize wandb
        wandb.init(project="max-activating-examples", config=vars(args))

        print(f"\nLoading crosscoder {args.crosscoder.split('/')[-2]}\n")
        crosscoder = load_crosscoder(args.crosscoder).to(
            device=args.cc_device, dtype=th.bfloat16)

        # TODO try first with all features
        # NOTE: looks like this is going to take too much storage
        # Get all feature indices from the crosscoder model
        # num_features = crosscoder.decoder.weight[1].shape[0]
        # selected_indices = list(range(num_features))
        # print(f"Using all {num_features} features from the crosscoder model\n")

        # df = load_latent_df("/share/u/troitskiid/projects/science-of-finetuning/results/eval_cro`sscoder/DeepScaleR-1.5B-crosscoder-L15-k100-lr1e-04-local-shuffling-CCLoss/data/feature_df.csv")
        # selected_features = df[(df["tag"].isin(
        #     ["IT only", "Base only", "ft only", "Reasoning only"]))]
        # selected_indices = selected_features.index.tolist()

        # L7

        l7_top_50_descending = [25456, 23188, 319, 9771, 7890, 8128, 2312, 25995, 30966, 9488, 30995, 21225, 24324, 5422, 10850, 23938, 23542, 29530, 18094, 14032, 29999, 10694, 5887,
                                27626, 8137, 31354, 7606, 30513, 10301, 10492, 350, 26691, 4968, 18143, 7780, 13265, 5486, 1492, 19161, 12320, 19025, 20870, 1942, 2814, 10354, 14358, 30854, 14938, 31870, 26342]

        l7_bottom_50_ascending = [10431, 761, 445, 15203, 8338, 621, 19563, 23517, 3379, 29250, 16182, 27527, 15757, 4597, 14711, 7807, 10645, 7533, 10673, 20330, 19167, 25632, 5045, 27113,
                                  4109, 22771, 25343, 13962, 31486, 13544, 17945, 11713, 7769, 13704, 26400, 16603, 28188, 10804, 18271, 16331, 16942, 30174, 27415, 6152, 24296, 21297, 31120, 2654, 3499, 25440]
        
        # Reverse the list to flip the order from last to first (so it's first top 50 and last bottom 50)
        l7_bottom_50_ascending = l7_bottom_50_ascending[::-1]

        l7_selected_indices = l7_top_50_descending + l7_bottom_50_ascending

        # L15

        l15_top_50_descending = [18832, 18663, 32732, 17615, 7510, 24996, 17455, 20781, 20197, 31660, 14122, 10256, 9725, 12819, 898, 4118, 25474, 31444, 31673, 17909, 28514, 12624, 21870,
                                 14040, 75, 636, 10499, 28974, 13415, 15907, 25158, 29751, 28081, 22295, 30322, 25362, 10635, 13899, 20580, 31748, 9702, 1452, 25548, 6273, 1199, 6185, 23278, 7142, 9233, 23142]

        l15_bottom_50_ascending = [243, 21616, 2840, 1565, 700, 22805, 4526, 32591, 30616, 16133, 31632, 5292, 5972, 8552, 26128, 3069, 17318, 12742, 23228, 32252, 7882, 10684, 30358,
                                   11145, 25929, 10024, 22897, 27474, 5217, 29985, 1451, 11556, 25878, 10080, 30520, 23274, 13670, 188, 7668, 11637, 31321, 25135, 30902, 701, 4199, 12194, 17393, 22658, 23400, 744]
        l15_bottom_50_ascending = l15_bottom_50_ascending[::-1]

        l15_selected_indices = l15_top_50_descending + l15_bottom_50_ascending

        # L23

        l23_top_50_descending = [3838, 212, 30150, 24763, 19686, 4675, 18533, 15648, 10170, 22495, 2586, 17897, 18659, 14078, 28594, 16872, 28859, 10227, 11372, 4356, 22913, 17896, 6578,
                                 12194, 13508, 29672, 11240, 14667, 5878, 7065, 26336, 32492, 17762, 18200, 3989, 12526, 1920, 1750, 7506, 2509, 24510, 26016, 20564, 13680, 9491, 22753, 25711, 32081, 32185, 11810]

        l23_bottom_50_ascending = [24105, 22239, 7314, 15968, 27874, 10914, 30616, 8556, 17913, 16145, 4481, 8978, 14362, 25421, 20183, 25716, 28857, 18892, 30855, 28577, 16262, 26580, 585,
                                   23960, 15789, 25782, 8034, 16307, 1977, 18559, 30946, 21161, 12508, 32286, 3517, 8180, 19502, 20025, 22104, 9581, 10636, 22368, 13104, 7865, 19825, 15518, 5383, 26983, 23423, 4733]
        l23_bottom_50_ascending = l23_bottom_50_ascending[::-1]
        
        l23_selected_indices = l23_top_50_descending + l23_bottom_50_ascending


        if args.layer == 7:
            selected_indices = l7_selected_indices
        elif args.layer == 15:
            selected_indices = l15_selected_indices
        elif args.layer == 23:
            selected_indices = l23_selected_indices


        # Load datasets
        test_set_base = load_dataset(
            "science-of-finetuning/fineweb-1m-sample", split="validation"
        )["text"]
        reasoning_column = "message_llama_chat_template"
        test_set_ft = load_dataset(
            "koyena/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B-formatted",
            split="validation",
        )[reasoning_column]
        test_set_base = test_set_base[: len(test_set_base) // 2]
        test_set_ft = test_set_ft[: len(test_set_ft) // 2]

        # Load models
        base_model = load_model(
            args.base_model,
            torch_dtype=th.bfloat16,
            attn_implementation="eager",
            dispatch=True,
            device_map=args.base_device,
        )
        ft_model = load_model(
            args.ft_model,
            torch_dtype=th.bfloat16,
            attn_implementation="eager",
            dispatch=True,
            device_map=args.ft_device,
        )

        # Create save directory if it doesn't exist
        save_path.mkdir(parents=True, exist_ok=True)

        # # Generate and save max activating examples
        # print("Generating mini examples...")
        # compute_max_activating_examples(
        #     test_set_ft[:100],
        #     selected_indices,
        #     crosscoder,
        #     model_batch_size=args.model_batch_size,
        #     crosscoder_batch_size=args.crosscoder_batch_size,
        #     n=args.n,
        #     base_model=base_model,
        #     ft_model=ft_model,
        #     layer=args.layer,
        #     cc_device=args.cc_device,
        #     workers=args.workers,
        #     max_seq_len=args.seq_len,
        #     save_path=save_path,
        #     checkpoint_every=50,
        #     name="mini-ft",
        # )

        print("Generating ft examples...")
        compute_max_activating_examples(
            test_set_ft,
            selected_indices,
            crosscoder,
            model_batch_size=args.model_batch_size,
            crosscoder_batch_size=args.crosscoder_batch_size,
            total_tokens=args.total_tokens,
            n=args.n,
            base_model=base_model,
            ft_model=ft_model,
            layer=args.layer,
            cc_device=args.cc_device,
            workers=args.workers,
            max_seq_len=args.seq_len,
            save_path=save_path,
            checkpoint_every=args.checkpoint_every,
            name="ft",
        )

        print("Generating base examples...")
        compute_max_activating_examples(
            test_set_base,
            selected_indices,
            crosscoder,
            model_batch_size=args.model_batch_size,
            crosscoder_batch_size=args.crosscoder_batch_size,
            total_tokens=args.total_tokens,
            n=args.n,
            base_model=base_model,
            ft_model=ft_model,
            layer=args.layer,
            cc_device=args.cc_device,
            workers=args.workers,
            max_seq_len=args.seq_len,
            save_path=save_path,
            checkpoint_every=args.checkpoint_every,
            name="base",
        )

        wandb.finish()
    # repo = df_hf_repo[args.crosscoder]
    ft_examples = th.load(save_path / "ft/ft_final.pt")
    base_examples = th.load(save_path / "base/base_final.pt")
    ft_base_examples = merge_max_examples(ft_examples, base_examples)
    th.save(ft_base_examples, save_path / "ft_base_examples.pt")
    max_act_exs_to_db(ft_base_examples, save_path / "ft_base_examples.db")

    # for file, file_name in [
    #     ("base/base_final", "base_examples"),
    #     ("ft/ft_final", "ft_examples"),
    #     ("ft_base_examples", "ft_base_examples"),
    # ]:
    #     for ftype in ["pt", "db"]:
    #         hf_api.upload_file(
    #             repo_id=repo,
    #             repo_type="dataset",
    #             path_or_fileobj=save_path / f"{file}.{ftype}",
    #             path_in_repo=f"{file_name}.{ftype}",
    #         )
    # create the ft_base_examples files


if __name__ == "__main__":
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
