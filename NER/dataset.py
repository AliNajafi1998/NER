import torch
import config


class EntityDataset:
    def __init__(self, texts, pos, tags):
        self.texts = texts
        self.pos = pos
        self.tags = tags

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        pos = self.pos[item]
        tag = self.tags[item]

        ids = []
        target_pos = []
        target_tag = []
        for i, s in enumerate(text):
            inputs = config.TOKENIZER.encode(
                s,
                add_special_tokens=False,
            )
            input_len = len(inputs)
            ids.extend(inputs)
            target_pos.extend([pos[i]] * input_len)
            target_tag.extend([tag[i]] * input_len)

        # truncating
        ids = ids[: config.MAX_LEN - 2]
        target_pos = target_pos[: config.MAX_LEN - 2]
        target_tag = target_tag[: config.MAX_LEN - 2]

        # adding special tokens
        ids = [101] + ids + [102]
        target_pos = [0] + target_pos + [0]
        target_tag = [0] + target_tag + [0]

        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        # padding
        padding_len = config.MAX_LEN - len(ids)
        ids = ids + [0] * padding_len
        mask = mask + [0] * padding_len
        target_pos = target_pos + [0] * padding_len
        target_tag = target_tag + [0] * padding_len
        token_type_ids = token_type_ids + [0] * padding_len

        return {
            "ids": torch.LongTensor(ids),
            "mask": torch.LongTensor(mask),
            "target_pos": torch.LongTensor(target_pos),
            "target_tag": torch.LongTensor(target_tag),
            "token_type_ids": torch.LongTensor(token_type_ids),
        }
