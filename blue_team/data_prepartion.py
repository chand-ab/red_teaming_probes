import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import Dataset, load_from_disk, load_dataset
from dotenv import load_dotenv
from baml_client import b, types
import fire

load_dotenv()


def create_sample_dataset():
    topics = [
        "philosophy",
        "art",
        "computer security",
        "sql injection",
        "geography",
        "martial arts",
        "power seeking",
        "productivity",
        "deceptive manipulation",
        "maths",
    ]

    data = {"question": [], "answer": [], "honest": []}

    def add_contrastive_pair(question: str, answer: bool):
        data["question"].append(question)
        data["answer"].append("Yes" if answer else "No")
        data["honest"].append(True)

        data["question"].append(question)
        data["answer"].append("No" if answer else "Yes")
        data["honest"].append(False)

    for topic in topics:
        result: types.Dataset = b.GenererateProbeDataset(topic=topic)

        for question in result.YesQuestions:
            add_contrastive_pair(question=question, answer=True)

        for question in result.NoQuestions:
            add_contrastive_pair(question=question, answer=False)

    dataset = Dataset.from_dict(data)
    dataset.save_to_disk("sleeper_agent_probe_dataset")

    return ""


def upload_to_hub():
    dataset = load_from_disk("sleeper_agent_probe_dataset")
    dataset.push_to_hub("poonai/sleeper_agent_probe")


def download_from_hub():
    dataset_path = Path("sleeper_agent_probe_dataset")

    if dataset_path.exists():
        shutil.rmtree(dataset_path)

    dataset = load_dataset("poonai/sleeper_agent_probe")
    dataset.save_to_disk("sleeper_agent_probe_dataset")

    return ""


if __name__ == "__main__":
    fire.Fire(
        {
            "create": create_sample_dataset,
            "upload": upload_to_hub,
            "download": download_from_hub,
        }
    )
