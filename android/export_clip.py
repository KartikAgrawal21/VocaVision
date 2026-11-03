"""
Exports CLIP (openai/clip-vit-base-patch32) for the Android app:
  * clip_image.onnx  - the image encoder, int8-quantised (about 90 MB), 224x224 -> 512-d embedding
  * clip_text.json   - pre-computed, normalised text embeddings for every label and colour prompt

Run from the repo root:  .venv\\Scripts\\python.exe android\\export_clip.py
"""
import json
import os
import sys

import numpy as np
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import clip_classifier as cc  # reuse the exact same prompts as the desktop app

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "src", "main", "assets")
MODEL_ID = "openai/clip-vit-base-patch32"

model = CLIPModel.from_pretrained(MODEL_ID).eval()
proc = CLIPProcessor.from_pretrained(MODEL_ID)


class ImageEncoder(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values):
        return self.m.get_image_features(pixel_values=pixel_values)


fp32 = os.path.join(ASSETS, "clip_image_fp32.onnx")
int8 = os.path.join(ASSETS, "clip_image.onnx")
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(ImageEncoder(model), dummy, fp32, input_names=["pixel_values"], output_names=["emb"],
                  opset_version=17, dynamo=False)
# Only MatMul is quantised: the Android ONNX Runtime has no ConvInteger kernel, so the
# patch-embedding Conv must stay float32 or the model fails to load on the phone.
quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])
os.remove(fp32)
for extra in (fp32 + ".data",):
    if os.path.exists(extra):
        os.remove(extra)
print("image encoder:", round(os.path.getsize(int8) / 1e6, 1), "MB")


def text_feats(prompts):
    with torch.no_grad():
        tok = proc(text=prompts, return_tensors="pt", padding=True)
        f = model.get_text_features(**tok)
        f = f / f.norm(dim=-1, keepdim=True)
    return f.numpy().round(5).tolist()


labels = [{"name": n, "kind": kind, "clothing": cls} for n, _, kind, cls in cc._LABELS]
data = {
    "labels": labels,
    "label_emb": text_feats([p for _, p, _, _ in cc._LABELS]),
    "colours": cc._COLOURS,
    "colour_emb": text_feats([f"a close-up of {c} fabric" for c in cc._COLOURS]),
}
with open(os.path.join(ASSETS, "clip_text.json"), "w") as fh:
    json.dump(data, fh)
print("labels:", len(labels), "colours:", len(cc._COLOURS))
