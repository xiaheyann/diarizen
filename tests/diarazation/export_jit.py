from pathlib import Path

import torch

from capriccio.inferencer.diarizen.inferencer import Inferencer
from capriccio.models.diarizen.model import Model as EENDModel
import json


def from_json(
    diarizen_hub: Path,
    device="cpu",
):
    config = json.load((diarizen_hub / "config.json").open("r", encoding="utf-8"))
    model = EENDModel(**config["model"]["args"])
    model.load_state_dict(
        torch.load(diarizen_hub / "pytorch_model.bin", map_location=device)
    )
    return model


if __name__ == "__main__":
    model = from_json(
        Path("model-bin/diarizen-wavlm-base"),
        device="cpu",
    )
    static_model = torch.jit.trace(model, torch.randn((1, 1, 256000)))
    # static_model = torch.jit.script(model)
    static_model.eval()
    frozen = torch.jit.freeze(static_model)
    torch.jit.save(frozen, "model-bin/diarizen-wavlm-base/jit_model.pt")
