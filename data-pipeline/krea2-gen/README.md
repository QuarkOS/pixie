# Krea2 3x3 @ 576 generation

Output and logs live in `./data` and `./logs` next to these scripts. Set
`KREA2_SPRITES` to the directory that holds `bin/sd-server`, `models/`, and
the generator virtualenv. Do not commit weights or generated sheets.

| Thing | Path / value |
|---|---|
| Raw 576 sheets (all kept) | `data/sheets/` |
| Sprites that pass every gate | `data/sprites32/ok/` |
| Structure OK, purity < 0.75 | `data/sprites32/purity_fail/` |
| Other extract failures | `data/sprites32/rejected/` |
| Labels (includes `edge_touch`) | `data/labels.jsonl` |
| Dashboard | http://127.0.0.1:8765 (loopback only) |
| Queue | 10,000 3x3 sheets / 90,000 sprites, seed 20261115 (~10 h at 2.47 sprites/s) |
| LoRA | `loras/p3-1500.safetensors` (hard link of the sd.cpp conversion of the 1500/final 3x3 checkpoint) |
| Models | `$KREA2_SPRITES/models` (read only) |

## Stop / resume

```bash
systemctl --user stop krea2-gen    # or: ./stop.sh
systemctl --user start krea2-gen   # resumes unfinished sheets
```

The runner skips sheets already recorded without error in `data/events.jsonl`. Raw sheets are never deleted.
