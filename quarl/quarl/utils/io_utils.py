import json

import torch


def print_rank_0(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def save_jsonl(data_lst, file_path, ensure_ascii=False, **kwargs):
    with open(file_path, "w") as f:
        f.write("\n".join([json.dumps(item, ensure_ascii=ensure_ascii, **kwargs) for item in data_lst]) + "\n")


def save_json(data, file_path, ensure_ascii=False, **kwargs):
    with open(file_path, "w") as f:
        json.dump(data, f, ensure_ascii=ensure_ascii, **kwargs)


def load_jsonl(file_path):
    with open(file_path, "r") as f:
        lines = f.read().strip().split("\n")
    return [json.loads(line) for line in lines]


def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    return data


if __name__ == "__main__":
    pass
