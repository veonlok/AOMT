"""Download the TextWorld (ALFWorld + WebShop) trajectory dataset from HuggingFace.

Dataset: Joshyxwa/cp2107-textworld-trajectories
Format:  Same JSONL+blocks format as ScienceWorld data — compatible with existing loader.
Access:  Gated dataset — requires HF account + agreeing to dataset conditions.
         Visit https://huggingface.co/datasets/Joshyxwa/cp2107-textworld-trajectories
         and click 'Agree and access repository' before running this script.

Usage:
    python fetch_textworld_data.py [--out_dir data/textworld]

After downloading, train with:
    python train_llada_rb.py --datasets scienceworld,textworld \\
        --data_dir data --data_dir_tw data/textworld ...
"""
import argparse, os, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/textworld")
    ap.add_argument("--token",   default=None,
                    help="HF access token (default: reads from ~/.cache/huggingface/token)")
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download, HfApi
    except ImportError:
        sys.exit("huggingface_hub not installed. Run: pip install huggingface_hub")

    # Verify token works
    api = HfApi(token=args.token)
    try:
        info = api.dataset_info("Joshyxwa/cp2107-textworld-trajectories", token=args.token)
        print(f"[hf] dataset found: {info.id}")
    except Exception as e:
        print(f"[hf] ERROR: {e}")
        print("\nIf you see 401/403: visit the dataset page and agree to conditions:")
        print("  https://huggingface.co/datasets/Joshyxwa/cp2107-textworld-trajectories")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[hf] downloading to {args.out_dir} ...")
    snapshot_download(
        repo_id="Joshyxwa/cp2107-textworld-trajectories",
        repo_type="dataset",
        local_dir=args.out_dir,
        token=args.token,
    )
    # Verify
    for split in ("train", "validation", "test"):
        path = os.path.join(args.out_dir, f"{split}.jsonl")
        if os.path.exists(path):
            n = sum(1 for _ in open(path) if _.strip())
            print(f"[ok] {split}.jsonl: {n} trajectories")
        else:
            print(f"[warn] {split}.jsonl not found at {path}")
    print("[done] textworld data ready. Train with --datasets textworld")

if __name__ == "__main__":
    main()
