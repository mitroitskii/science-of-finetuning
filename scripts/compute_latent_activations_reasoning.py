import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from transformers import AutoTokenizer
from tqdm.auto import trange
import torch as th
from argparse import ArgumentParser

from tools.paths import *
from tools.utils import load_activation_dataset, load_dictionary_model
from tools.cc_utils import chat_only_latent_indices, base_only_latent_indices, shared_latent_indices


# NOTE: couldn't run this time because I cached my activations without the BOS token; make sure to cache with BOS for this to work

@th.no_grad()
def get_positive_activations(sequences, ranges, dataset, cc, latent_ids):
    """
    Extract positive activations and their indices from sequences.
    Also compute the maximum activation for each latent feature.

    Args:
        sequences: List of sequences
        ranges: List of (start_idx, end_idx) tuples for each sequence
        dataset: Dataset containing activations
        cc: Object with get_activations method
        latent_ids: Tensor of latent indices to extract

    Returns:
        Tuple of:
        - activations tensor: positive activation values
        - indices tensor: in (seq_idx, seq_pos, feature_pos) format
        - seq_ranges tensor: cumulative count of positive activations per sequence end
        - max_activations: maximum activation value for each latent feature
    """
    out_activations = []
    out_ids = []
    # seq_ranges_list[k] stores the total number of positive activations found
    # up to the end of sequence k-1. seq_ranges_list[0] is 0.
    seq_ranges_list = [0]
    current_total_pos_activations = 0

    if len(latent_ids) == 0:
        raise ValueError(
            "latent_ids cannot be empty. No latent features were selected or provided.")

    max_activations = th.zeros(len(latent_ids), device='cuda')

    for seq_idx in trange(len(sequences), desc="Processing sequences"):
        start_idx, end_idx = ranges[seq_idx]
        # Skip if range is invalid or empty
        if start_idx >= end_idx or start_idx >= len(dataset) or end_idx > len(dataset):
            print(
                f"Warning: Invalid range ({start_idx}, {end_idx}) for sequence {seq_idx} (dataset size {len(dataset)}). Skipping.")
            # No activations added
            seq_ranges_list.append(current_total_pos_activations)
            continue

        activations = th.stack([dataset[j].cuda()
                               for j in range(start_idx, end_idx)])
        # Skip if no activations were loaded (e.g., empty range resulted in empty stack)
        if activations.numel() == 0:
            print(
                f"Warning: No activations loaded for sequence {seq_idx} (range {start_idx}-{end_idx}). Skipping.")
            # No activations added
            seq_ranges_list.append(current_total_pos_activations)
            continue

        feature_activations = cc.get_activations(activations)
        assert feature_activations.shape == (len(activations), len(latent_ids))

        # Track maximum activations
        if feature_activations.numel() > 0:  # Ensure not empty before max
            # For each latent feature, find the max activation in this sequence
            seq_max_values, _ = feature_activations.max(dim=0)
            # Update global maximums where this sequence has a higher value
            update_mask = seq_max_values > max_activations
            max_activations[update_mask] = seq_max_values[update_mask]

        # Get indices where feature activations are positive
        pos_mask = feature_activations > 0
        pos_indices = th.nonzero(pos_mask, as_tuple=True)
        pos_activations = feature_activations[pos_mask]

        num_pos_in_seq = pos_activations.numel()

        if num_pos_in_seq > 0:
            # Create sequence indices tensor matching size of positive indices
            seq_idx_tensor = th.full_like(pos_indices[0], seq_idx)
            # Stack indices into (seq_idx, seq_pos, feature_pos) format
            pos_ids = th.stack(
                [seq_idx_tensor, pos_indices[0], pos_indices[1]], dim=1)

            out_activations.append(pos_activations)
            out_ids.append(pos_ids)

        # Update cumulative count for seq_ranges
        current_total_pos_activations += num_pos_in_seq
        seq_ranges_list.append(current_total_pos_activations)

    # Convert seq_ranges list to tensor AFTER the loop
    seq_ranges_tensor = th.tensor(seq_ranges_list, dtype=th.int64).cpu()

    # Handle case where no positive activations were found across all sequences
    if not out_activations:
        print(
            f"Warning: No positive activations found across {len(sequences)} sequences.")
        # Return empty tensors for activations and indices
        empty_activations = th.tensor([], dtype=th.float32).cpu()
        empty_ids = th.tensor([], dtype=th.int64).reshape(0, 3).cpu()

        # max_activations might still contain valid data (non-positive maxes). Return it.
        return empty_activations, empty_ids, seq_ranges_tensor, max_activations.cpu()

    # Concatenate results if positive activations were found
    out_activations_tensor = th.cat(out_activations).cpu()
    out_ids_tensor = th.cat(out_ids).cpu()

    return out_activations_tensor, out_ids_tensor, seq_ranges_tensor, max_activations.cpu()


def split_into_sequences(tokenizer, tokens):
    # Find indices of BOS tokens
    indices_of_bos = th.where(tokens == tokenizer.bos_token_id)[0]

    # Split tokens into sequences starting with BOS token
    sequences = []
    index_to_seq_pos = []  # List of (sequence_idx, idx_in_sequence) tuples
    ranges = []
    for i in trange(len(indices_of_bos)):
        start_idx = indices_of_bos[i]
        end_idx = indices_of_bos[i +
                                 1] if i < len(indices_of_bos)-1 else len(tokens)
        sequence = tokens[start_idx:end_idx]
        sequences.append(sequence)
        ranges.append((start_idx, end_idx))
        # Add mapping for each token in this sequence
        for j in range(len(sequence)):
            index_to_seq_pos.append((i, j))

    return sequences, index_to_seq_pos, ranges


# FIXME change the repo id
def load_latent_activations(repo_id="science-of-finetuning/autointerp-data-gemma-2-2b-l13-mu4.1e-02-lr1e-04"):
    """
    Load the autointerp data from Hugging Face Hub.

    Args:
        repo_id (str): The Hugging Face Hub repository ID containing the data

    Returns:
        tuple: (activations, indices, sequences) tensors where:
            - activations: tensor of shape [n_total_activations] containing latent activations
            - indices: tensor of shape [n_total_activations, 3] containing (seq_idx, seq_pos, latent_idx)
            - sequences: tensor of shape [n_total_sequences, max_seq_len] containing the padded input sequences (right padded)
    """
    import torch
    from huggingface_hub import hf_hub_download

    # Download files from hub
    activations_path = hf_hub_download(
        repo_id=repo_id, filename="activations.pt", repo_type="dataset")
    indices_path = hf_hub_download(
        repo_id=repo_id, filename="indices.pt", repo_type="dataset")
    sequences_path = hf_hub_download(
        repo_id=repo_id, filename="sequences.pt", repo_type="dataset")
    latent_ids_path = hf_hub_download(
        repo_id=repo_id, filename="latent_ids.pt", repo_type="dataset")

    # Load tensors
    activations = torch.load(activations_path, weights_only=False)
    indices = torch.load(indices_path, weights_only=False)
    sequences = torch.load(sequences_path, weights_only=False)
    latent_ids = torch.load(latent_ids_path, weights_only=False)

    return activations, indices, sequences, latent_ids


def main():
    parser = ArgumentParser(
        description='Compute positive and maximum activations for latent features')
    parser.add_argument("--activation-store-dir", type=str,
                        default="~/data/activations/")
    parser.add_argument("--indices-root", type=str,
                        default="~/data/latent_indices/")
    parser.add_argument("--target-set", type=str, nargs='+', default=[])
    parser.add_argument("--base-model", type=str,
                        default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--ft-model", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--dictionary-model", type=str, required=True)
    parser.add_argument("--latent-activations-dir", type=str,
                        default="~/data/latent_activations/")
    parser.add_argument("--upload-to-hub", action="store_true")
    parser.add_argument("--from-hub", action="store_true")
    parser.add_argument("--split", type=str, default="validation")
    args = parser.parse_args()

    # Load the activation dataset
    if not args.from_hub:
        fineweb_cache, magpie_cache = load_activation_dataset(
            activation_store_dir=args.activation_store_dir,
            base_model=args.base_model.split("/")[-1],
            instruct_model=args.ft_model.split("/")[-1],
            layer=args.layer,
            lmsys_name="Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B-formatted",
            fineweb_name="fineweb-1m-sample",
            lmsys_split="validation-col-message_llama_chat_template",
            fineweb_split="validation",
        )
        tokens_fineweb = fineweb_cache.tokens[0]
        tokens_magpie = magpie_cache.tokens[0]
        print(f"Loaded {tokens_fineweb.numel()} FineWeb tokens.")
        print(f"Loaded {tokens_magpie.numel()} Magpie tokens.")

        # Load the dictionary model
        print(f"Loading dictionary model: {args.dictionary_model}")
        dictionary_model = load_dictionary_model(
            args.dictionary_model).to(device="cuda", dtype=th.bfloat16)

        # Load the tokenizer
        print(f"Loading tokenizer for: {args.ft_model}")
        tokenizer = AutoTokenizer.from_pretrained(args.ft_model)

        print("Splitting tokens into sequences...")
        seq_magpie, _, ranges_magpie = split_into_sequences(
            tokenizer, tokens_magpie)
        seq_fineweb, _, ranges_fineweb = split_into_sequences(
            tokenizer, tokens_fineweb)
        print(
            f"Split into {len(seq_fineweb)} FineWeb sequences and {len(seq_magpie)} Magpie sequences.")

        indices_root = Path(args.indices_root)
        if len(args.target_set) == 0:
            print("Using all latent features.")
            latent_ids = th.arange(dictionary_model.dict_size)
        else:
            print(f"Loading target latent indices from: {args.target_set}")
            indices = []
            for target_set in args.target_set:
                fpath = indices_root / f"{target_set}.pt"
                print(f"Loading {fpath}...")
                indices.append(th.load(fpath, weights_only=True))
            latent_ids = th.cat(indices)

        print(f"Selected {len(latent_ids)} latent features.")
        # Move selected ids to cuda for get_activations
        latent_ids = latent_ids.to('cuda')

        print("\nProcessing FineWeb activations...")
        out_acts_fineweb, out_ids_fineweb, seq_ranges_fineweb, max_activations_fineweb = get_positive_activations(
            seq_fineweb, ranges_fineweb, fineweb_cache, dictionary_model, latent_ids
        )
        print(
            f"Found {out_acts_fineweb.numel()} positive activations in FineWeb.")

        print("\nProcessing Magpie activations...")
        out_acts_magpie, out_ids_magpie, seq_ranges_magpie, max_activations_magpie = get_positive_activations(
            seq_magpie, ranges_magpie, magpie_cache, dictionary_model, latent_ids
        )
        print(
            f"Found {out_acts_magpie.numel()} positive activations in Magpie.")

        # --- Combine Results ---
        print("\nCombining results...")

        # Combine activations
        out_acts = th.cat([out_acts_fineweb, out_acts_magpie])

        # Adjust Magpie sequence indices before concatenating IDs
        # Only adjust if there are Magpie indices to adjust and fineweb sequences exist
        if out_ids_magpie.numel() > 0 and len(seq_fineweb) > 0:
            # Offset by the number of fineweb sequences
            out_ids_magpie[:, 0] += len(seq_fineweb)

        # Combine indices
        out_ids = th.cat([out_ids_fineweb, out_ids_magpie])

        # Adjust Magpie seq_ranges before concatenating
        # The offset should be the total count of positive activations from fineweb.
        # This is the last value in seq_ranges_fineweb.
        # seq_ranges_* tensors include the initial 0, so shape is [num_seq + 1]
        fineweb_total_pos_count = seq_ranges_fineweb[-1].item(
        ) if seq_ranges_fineweb.numel() > 0 else 0

        # Calculate adjusted part of magpie ranges (all elements except the first 0)
        # Only adjust if there are magpie ranges beyond the initial 0
        if seq_ranges_magpie.numel() > 1:
            adjusted_seq_ranges_magpie = seq_ranges_magpie[1:] + \
                fineweb_total_pos_count
        else:
            # If magpie had no sequences or no activations, seq_ranges_magpie is just [0]
            # So the adjusted part is empty
            adjusted_seq_ranges_magpie = th.tensor([], dtype=th.int64)

        # Concatenate seq_ranges: take all of fineweb, and the adjusted part of magpie
        seq_ranges = th.cat([seq_ranges_fineweb, adjusted_seq_ranges_magpie])

        # Combine max activations, taking the element-wise maximum
        # Ensure tensors are on the same device (should both be CPU from function return)
        # Handle case where one might be empty if latent_ids was empty (though checked earlier)
        if max_activations_fineweb.numel() > 0 and max_activations_magpie.numel() > 0:
            combined_max_activations = th.maximum(
                max_activations_fineweb, max_activations_magpie)
        elif max_activations_fineweb.numel() > 0:
            combined_max_activations = max_activations_fineweb
        elif max_activations_magpie.numel() > 0:
            combined_max_activations = max_activations_magpie
        else:
            # Should not happen if latent_ids not empty
            combined_max_activations = th.tensor([], dtype=th.float32)

        # --- Padding and Saving ---
        print("Padding sequences...")
        sequences_all = seq_fineweb + seq_magpie

        # Check if sequences_all is empty before proceeding
        if not sequences_all:
            print(
                "Warning: No sequences found in either dataset. Skipping padding and saving sequences.")
            padded_tensor = th.tensor(
                [], dtype=th.int64)  # Or appropriate dtype
            seq_lengths = th.tensor([], dtype=th.int64)
        else:
            max_len = max(len(s) for s in sequences_all)
            seq_lengths = th.tensor([len(s) for s in sequences_all])
            # Pad each sequence to max length
            # Ensure sequences are on CPU before padding if they aren't already
            padded_seqs = [th.cat([s.cpu(), th.full(
                (max_len - len(s),), tokenizer.pad_token_id, dtype=s.dtype)]) for s in sequences_all]
            padded_tensor = th.stack(padded_seqs)

        # Extract the last two sections before the last '/'
        dict_model_path = str(args.dictionary_model)
        path_parts = dict_model_path.split('/')
        if len(path_parts) >= 3:
            model_identifier = f"{path_parts[-3]}/{path_parts[-2]}"
        else:
            model_identifier = '/'.join(path_parts[:-1]) if len(path_parts) > 1 else path_parts[0]
        
        out_dir = Path(args.latent_activations_dir) / model_identifier
        print(f"Saving computed tensors to: {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save tensors
        th.save(out_acts.cpu(), out_dir / "out_acts.pt")
        th.save(out_ids.cpu(), out_dir / "out_ids.pt")
        th.save(padded_tensor.cpu(), out_dir / "padded_sequences.pt")
        th.save(latent_ids.cpu(), out_dir / "latent_ids.pt")
        th.save(seq_ranges.cpu(), out_dir / "seq_ranges.pt")
        th.save(seq_lengths.cpu(), out_dir / "seq_lengths.pt")
        th.save(combined_max_activations.cpu(), out_dir / "max_activations.pt")
        print("Finished saving tensors.")

        # --- Print Stats --- (Only when calculating)
        print(f"\nCombined maximum activation statistics:")
        print(f"  Number of features: {combined_max_activations.numel()}")
        if combined_max_activations.numel() > 0:
            print(f"  Average: {combined_max_activations.mean().item():.4f}")
            print(f"  Maximum: {combined_max_activations.max().item():.4f}")
            print(f"  Minimum: {combined_max_activations.min().item():.4f}")
        else:
            print("  No max activations recorded.")

    if args.upload_to_hub:
        # Initialize Hugging Face API
        from huggingface_hub import HfApi
        api = HfApi()

        # Define repository ID for the dataset
        repo_id = f"science-of-finetuning/latent-activations-{args.dictionary_model}"

        # Upload all tensors to HF Hub directly from saved files
        api.upload_file(
            path_or_fileobj=str(out_dir / "out_acts.pt"),
            path_in_repo="activations.pt",
            repo_id=repo_id,
            repo_type="dataset",
        )

        api.upload_file(
            path_or_fileobj=str(out_dir / "out_ids.pt"),
            path_in_repo="indices.pt",
            repo_id=repo_id,
            repo_type="dataset"
        )

        api.upload_file(
            path_or_fileobj=str(out_dir / "padded_sequences.pt"),
            path_in_repo="sequences.pt",
            repo_id=repo_id,
            repo_type="dataset"
        )

        api.upload_file(
            path_or_fileobj=str(out_dir / "latent_ids.pt"),
            path_in_repo="latent_ids.pt",
            repo_id=repo_id,
            repo_type="dataset"
        )

        # Upload max activations and indices
        api.upload_file(
            path_or_fileobj=str(out_dir / "max_activations.pt"),
            path_in_repo="max_activations.pt",
            repo_id=repo_id,
            repo_type="dataset"
        )

        print(f"All files uploaded to Hugging Face Hub at {repo_id}")


if __name__ == "__main__":
    main()
