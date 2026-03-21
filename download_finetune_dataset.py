from datasets import load_dataset
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Download anthrop_nvidiasteer dataset from Hugging Face"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./anthrop_nvidiasteer_dataset",
        help="Output directory for dataset",
    )
    args = parser.parse_args()

    print(f"Loading dataset 'achand45/anthrop_nvidiasteer'...")
    ds = load_dataset("achand45/anthrop_nvidiasteer")
    print(f"Dataset loaded: {ds}")

    print(f"Saving dataset to '{args.output}'...")
    ds.save_to_disk(args.output)
    print(f"Dataset saved successfully to '{args.output}'")


if __name__ == "__main__":
    main()
