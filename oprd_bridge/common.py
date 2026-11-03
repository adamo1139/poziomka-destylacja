"""Wspolne narzedzia OPRD-Bridge: JSONL, szablony czatu, ladowanie modeli, haki na warstwy."""

import json
import os
from pathlib import Path

# W srodowisku ENTER siedzi TF 2.15 z protobufem 7 - import modelingu przez
# transformers wywala sie na nim (MessageFactory.GetPrototype). TF nie jest nam potrzebny.
os.environ.setdefault("USE_TF", "0")

import torch


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, rows):
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def wczytaj_dotenv(*paths):
    """Uzupelnia os.environ z plikow .env (prawdziwa zmienna ma pierwszenstwo)."""
    for path in paths:
        path = Path(path)
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def render_prompt_ids(tokenizer, messages, thinking=None):
    """Prompt z generation promptem jako tokeny - tak samo jak chat/chat_poziomka.py.

    thinking=None nie przekazuje enable_thinking (szablony bez tej opcji).
    """
    kwargs = {} if thinking is None else {"enable_thinking": thinking}
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **kwargs
    )
    return tokenizer.encode(text, add_special_tokens=False)


def check_same_vocab(tok_a, tok_b):
    """OPRD bez retokenizacji wymaga identycznego slownika (u nas APT4)."""
    if tok_a.get_vocab() != tok_b.get_vocab():
        raise ValueError("Tokenizery ucznia i nauczyciela maja rozne slowniki - ten kod zaklada wspolny.")


def load_model(path, device, dtype=torch.bfloat16, attn="sdpa"):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=dtype, attn_implementation=attn, trust_remote_code=True
    )
    model.to(device).eval()
    model.requires_grad_(False)
    return model


def decoder_layers(model):
    """Bloki transformera bez warstw MTP (Bailing trzyma je na koncu self.layers)."""
    layers = model.get_decoder().layers
    return list(layers)[: model.config.num_hidden_layers]


def layer_map(num_student, num_teacher):
    """phi z rownania (11) w wersji 0-based, identycznie jak oficjalny kod (linspace + round)."""
    if num_student == num_teacher:
        return list(range(num_student))
    if num_student == 1:
        return [num_teacher - 1]
    return torch.linspace(0, num_teacher - 1, steps=num_student).round().long().tolist()


class LayerCapture:
    """Zbiera wyjscia wskazanych blokow (residual stream PRZED koncowym normem).

    HF hidden_states[-1] ma juz nalozony model.norm, wiec ostatnia warstwa bylaby
    nieporownywalna z reszta - stad haki zamiast output_hidden_states.
    """

    def __init__(self, model, indices):
        self.layers = decoder_layers(model)
        self.indices = sorted(set(indices))
        self.outputs = {}
        self._handles = []

    def __enter__(self):
        for idx in self.indices:
            self._handles.append(self.layers[idx].register_forward_hook(self._hook(idx)))
        return self

    def _hook(self, idx):
        def hook(_module, _inputs, output):
            self.outputs[idx] = output[0] if isinstance(output, tuple) else output
        return hook

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
