"""
download the fineweb pre-training corpus

Loads only the first ``num_samples`` rows (``train[:N]``), then shuffles that
subset. This avoids materializing the full sample-10BT split. It is not
statistically identical to ``shuffle(full_train).select(N)`` (uniform draw from
the entire pool).

```
python scripts/data_process/fineweb.py --num_samples=10000000 --min_length_for_memory=2048 --validation_size=3000
```
"""

import os
from typing import Any, Dict, List

from absl import app, flags
from datasets import load_dataset
from transformers import AutoTokenizer

FLAGS = flags.FLAGS


def set_args():
    flags.DEFINE_integer(
        "num_samples",
        default=10_000_000,
        help="number of samples to sample from the FineWeb.",
    )
    flags.DEFINE_integer(
        "min_length_for_memory",
        default=2048,
        help="minimum length for pre-training text subset with memory.",
    )
    flags.DEFINE_integer(
        "validation_size",
        default=3_000,
        help="number of samples for validation set.",
    )
    flags.DEFINE_integer(
        "num_proc",
        0,
        help="Parallel workers for map/filter/save_to_disk; 0 = min(CPU count, 128).",
    )
    flags.DEFINE_integer(
        "map_batch_size",
        1024,
        help="Batch size for tokenization map.",
    )


def _num_proc() -> int:
    if FLAGS.num_proc > 0:
        return FLAGS.num_proc
    return max(1, min(os.cpu_count() or 1, 128))


def main(argv):
    tokenizer = AutoTokenizer.from_pretrained("alpindale/Llama-3.2-1B-Instruct")
    num_samples = FLAGS.num_samples
    num_proc = _num_proc()

    # Efficient subset: only Arrow shards for train[:N] are read, not the full 10BT split.
    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split=f"train[:{num_samples}]",
    )
    random_seed = 42
    dataset = dataset.shuffle(seed=random_seed)

    total_samples = len(dataset)
    print("total samples num:", total_samples)

    def tokenize_texts(examples: Dict[str, List[Any]]):
        token_counts = tokenizer(
            examples["text"], add_special_tokens=False
        )["input_ids"]
        examples["num_tokens"] = [len(x) for x in token_counts]
        return examples

    dataset_with_token_num = dataset.map(
        tokenize_texts,
        batched=True,
        batch_size=FLAGS.map_batch_size,
        num_proc=num_proc,
    )

    def filter_fn(examples: Dict[str, List[Any]]):
        token_counts = examples["num_tokens"]
        return [x > FLAGS.min_length_for_memory for x in token_counts]

    filtered_dataset = dataset_with_token_num.filter(
        filter_fn, batched=True, num_proc=num_proc
    )
    filtered_dataset = filtered_dataset.remove_columns("num_tokens")

    n_f = len(filtered_dataset)
    text_mem = filtered_dataset.select(range(0, n_f // 2))
    text_inst = filtered_dataset.select(range(n_f // 2, n_f))

    text = filtered_dataset.train_test_split(test_size=FLAGS.validation_size)
    text_mem = text_mem.train_test_split(test_size=FLAGS.validation_size)
    text_inst = text_inst.train_test_split(test_size=FLAGS.validation_size)

    print(
        "text train:",
        len(text["train"]),
        "text test:",
        len(text["test"]),
        "text_mem train:",
        len(text_mem["train"]),
        "text_mem test:",
        len(text_mem["test"]),
        "text_inst train:",
        len(text_inst["train"]),
        "text_inst test:",
        len(text_inst["test"]),
    )
    shards = {"train": 128, "test": 4}
    save_proc = min(num_proc, 128)
    text.save_to_disk(
        "dataset_cache/processed/fineweb/text",
        num_shards=shards,
        num_proc=save_proc,
    )
    text_mem.save_to_disk(
        "dataset_cache/processed/fineweb/text_mem",
        num_shards=shards,
        num_proc=save_proc,
    )
    text_inst.save_to_disk(
        "dataset_cache/processed/fineweb/text_inst",
        num_shards=shards,
        num_proc=save_proc,
    )


if __name__ == "__main__":
    set_args()
    app.run(main)
