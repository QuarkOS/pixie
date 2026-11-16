"""Calculate model tensor sizes and explicitly estimated run costs without assets."""
import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.pixie32.p32_model import P32, P32Config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with torch.device("meta"):
        model = P32(P32Config())
    parameters = sum(p.numel() for p in model.parameters())
    weights = sum(p.numel() * p.element_size() for p in model.parameters())
    state = sum(p.numel() * p.element_size() for p in model.state_dict().values())
    report = {
        "source_shape_calculation": {
            "parameters": parameters, "fp32_parameter_bytes": weights,
            "state_tensor_bytes": state, "no_weights_or_gpu_loaded": True,
        },
        "estimates": {
            "model_plus_two_fp32_adam_moments_bytes": state + 2 * weights,
            "model_snapshots_at_100k_over_12m": 120,
            "snapshot_tensor_payload_bytes_at_100k_over_12m": 120 * state,
            "views_12m_divided_by_r1_count_133688": 12_000_000 / 133_688,
            "training_only_hours_at_supplied_estimate_357_img_s": 12_000_000 / 357 / 3600,
            "remaining_training_only_hours_from_694528_views_at_357_img_s": (12_000_000 - 694_528) / 357 / 3600,
            "seconds_per_100k_views_at_357_img_s": 100_000 / 357,
            "scope": "arithmetic estimates; exclude checkpoint serialization, reporting, startup and phase variability; R1 ratio is not weighted train epochs",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
