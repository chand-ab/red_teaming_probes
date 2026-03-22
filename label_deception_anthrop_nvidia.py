import sys
from pathlib import Path
from tqdm import tqdm
import time
from functools import wraps

sys.path.insert(0, str(Path(__file__).parent))

from datasets import load_from_disk, load_dataset, Dataset
from dotenv import load_dotenv
from baml_client import b
import fire

load_dotenv()


def retry_with_timeout(max_retries=10, initial_timeout=2, max_timeout=60):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_count = 0
            current_timeout = initial_timeout

            while retry_count < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        print(f"Max retries ({max_retries}) reached. Error: {e}")
                        raise

                    wait_time = min(current_timeout, max_timeout)
                    print(
                        f"Timeout/Error: {e}. Retrying in {wait_time}s (attempt {retry_count}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    current_timeout *= 2

            return None

        return wrapper

    return decorator


def label_anthrop_nvidia_deception(batch_size=100, num_samples=None):
    dataset = load_from_disk("anthrop_nvidiasteer_dataset")
    train_data = dataset["train"]

    if num_samples is not None:
        train_data = train_data.select(range(min(num_samples, len(train_data))))
        print(f"Testing with {len(train_data)} samples")
    else:
        print(f"Processing all {len(train_data)} samples")

    assistant_contents = []
    all_messages = []

    for idx in tqdm(range(len(train_data)), desc="Extracting assistant responses"):
        messages = train_data[idx]["messages"]
        all_messages.append(messages)

        assistant_content = None
        for msg in messages:
            if msg["role"] == "assistant":
                assistant_content = msg["content"]
                break

        assistant_contents.append(assistant_content if assistant_content else "")

    deception_labels = [False] * len(assistant_contents)
    num_batches = (len(assistant_contents) + batch_size - 1) // batch_size

    @retry_with_timeout(max_retries=5, initial_timeout=2, max_timeout=60)
    def label_batch(batch_contents):
        result = b.LabelDeceptionBatch(targets=batch_contents)
        return result.labels

    for batch_idx in tqdm(range(num_batches), desc="Labeling batches"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(assistant_contents))
        batch_contents = assistant_contents[start_idx:end_idx]

        batch_labels = label_batch(batch_contents)
        actual_batch_size = end_idx - start_idx
        deception_labels[start_idx:end_idx] = batch_labels[:actual_batch_size]

    labeled_data = {
        "messages": all_messages,
        "assistant_content": assistant_contents,
        "is_deceptive": deception_labels,
    }

    labeled_dataset = Dataset.from_dict(labeled_data)
    labeled_dataset.save_to_disk("anthrop_nvidiasteer_labeled")

    print(f"\nLabeled dataset saved to anthrop_nvidiasteer_labeled")
    print(f"Total samples: {len(labeled_dataset)}")
    print(f"Deceptive samples: {sum(deception_labels)}")
    print(f"Non-deceptive samples: {len(deception_labels) - sum(deception_labels)}")

    return ""


def push_to_hub(
    dataset_path="anthrop_nvidiasteer_labeled", repo_id="poonai/deceptive_labled"
):
    dataset = load_from_disk(dataset_path)
    dataset.push_to_hub(repo_id)
    print(f"Dataset pushed to {repo_id}")
    return ""


def download_from_hub(
    repo_id="poonai/deceptive_labled", dataset_path="anthrop_nvidiasteer_labeled"
):
    from pathlib import Path
    import shutil

    dataset_path_obj = Path(dataset_path)

    if dataset_path_obj.exists():
        shutil.rmtree(dataset_path_obj)
        print(f"Removed existing directory: {dataset_path}")

    dataset = load_dataset(repo_id)
    dataset.save_to_disk(dataset_path)
    print(f"Dataset downloaded from {repo_id} and saved to {dataset_path}")
    print(f"Total samples: {len(dataset)}")
    return ""


if __name__ == "__main__":
    fire.Fire(
        {
            "label": label_anthrop_nvidia_deception,
            "push": push_to_hub,
            "download": download_from_hub,
        }
    )
