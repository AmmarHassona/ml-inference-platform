import json
from pathlib import Path
from datasets import load_dataset

SAMPLES_PER_CLASS = 15

LABEL_MAP = {
    0: "world",
    1: "sports",
    2: "business",
    3: "sci_tech",
}

def main():
    print("Loading AG News...")
    ds = load_dataset("ag_news", split="train")

    corpus = []
    counts = {label: 0 for label in LABEL_MAP.values()}

    for example in ds:
        label = LABEL_MAP[example["label"]]
        if counts[label] >= SAMPLES_PER_CLASS:
            continue
        corpus.append({"text": example["text"].split("\\n")[0].strip(), "label": label})
        counts[label] += 1
        if all(c >= SAMPLES_PER_CLASS for c in counts.values()):
            break

    out_path = Path(__file__).parent.parent / "app" / "corpus.json"
    with open(out_path, "w") as f:
        json.dump(corpus, f, indent=2)

    print(f"Written {len(corpus)} entries to {out_path}")
    for label, count in counts.items():
        print(f"  {label}: {count}")

if __name__ == "__main__":
    main()