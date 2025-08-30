import os
import json
from typing import List, Optional, Dict, Sequence, Tuple
from transformers import PreTrainedTokenizer
import pandas as pd

class colonTokenizer(PreTrainedTokenizer):

    model_input_names = ["input_ids"]


    def __init__(self,
                 model_max_length: int,
                 characters: Sequence[str] = ("A", "C", "G", "T"),
                 pad_token="N",
                 **kwargs):


        self.characters = characters
        self.model_max_length = model_max_length

        # === 特殊 token 列表，确保 [PAD] 是 ID=0 ===
        self.special_tokens = [pad_token]
        self.base_tokens = list(characters)

        # === vocab 构建 ===
        self._vocab_str_to_int = {
            **{tok: i for i, tok in enumerate(self.special_tokens)},
            **{b: len(self.special_tokens) + i for i, b in enumerate(self.base_tokens)}
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}

        super().__init__(
            pad_token=pad_token,
            model_max_length=model_max_length,
            **kwargs
        )


    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def __len__(self):
        return self.vocab_size

    def _tokenize(self, text: str) -> List[str]:
        return list(text.upper())

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int[self.pad_token])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str.get(index, self.pad_token)

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self._convert_id_to_token(i) for i in ids]

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return "".join([t for t in tokens if len(t) == 1 and t in self.base_tokens])

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        return token_ids_0 + ([] if token_ids_1 is None else token_ids_1)

    def get_vocab(self) -> Dict[str, int]:
        return self._vocab_str_to_int

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        path = os.path.join(save_directory, (filename_prefix or "") + "vocab.json")
        with open(path, "w") as f:
            json.dump(self._vocab_str_to_int, f)
        return (path,)

    @classmethod
    def from_pretrained(cls, dir_path: str, **kwargs):
        path = os.path.join(dir_path, "vocab.json")
        with open(path, "r") as f:
            vocab = json.load(f)

        id_to_str = {v: k for k, v in vocab.items()}
        characters = [k for k in vocab.keys() if len(k) == 1 and k.isalpha()]
        tokenizer = cls(
            model_max_length=kwargs.get("model_max_length", 512),
            characters=characters
        )
        tokenizer._vocab_str_to_int = vocab
        tokenizer._vocab_int_to_str = id_to_str
        return tokenizer

    def __call__(self, text, **kwargs):

        # Tokenize to base tokens only
        tokens = self._tokenize(text)
        input_ids = [self._convert_token_to_id(tok) for tok in tokens]
        
        # mask: 只要不是pad_token就为1
        attention_mask = [0 if tok == self.pad_token else 1 for tok in tokens]

        # === 截断 ===
        if len(input_ids) > self.model_max_length:
            input_ids = input_ids[:self.model_max_length]
            attention_mask = attention_mask[:self.model_max_length]
            
        # padding
        while len(input_ids) < self.model_max_length:
            input_ids.append(self._convert_token_to_id(self.pad_token))
            attention_mask.append(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

