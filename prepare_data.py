import os
import numpy as np
import tiktoken
from datasets import load_dataset

enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token                     # 50256: the "end of text" token


def tokenize(example):
    ids = enc.encode_ordinary(example["text"])   # text → list of token IDs
    ids.append(EOT)                              # mark where each story ends
    return {"ids": ids, "len": len(ids)}


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    ds = load_dataset("roneneldan/TinyStories")  # splits: "train", "validation"

    for split in ["train", "validation"]:
        tok = ds[split].map(tokenize, remove_columns=["text"], num_proc=4,
                            desc=f"tokenizing {split}")
        total = int(np.sum(tok["len"], dtype=np.uint64))
        path = f"data/tinystories_{split}.bin"
        arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(total,))

        pos = 0
        n_shards = 64
        for i in range(n_shards):
            shard = tok.shard(num_shards=n_shards, index=i, contiguous=True).with_format("numpy")
            chunk = np.concatenate(shard["ids"])
            arr[pos : pos + len(chunk)] = chunk
            pos += len(chunk)
        arr.flush()
        print(f"{split}: {total:,} tokens → {path}")