# Krea 2 32x32 sheet LoRA: dataset + training plan (box build 2026-11-14)

## Dataset
- 840 real 32x32 sprites (no AI-generated, no upscaled), split 798 train / 42 eval (5%, seed 42).
  - sprites/NNNN.png (32x32 RGBA), sprites/meta.json (src, original path, label, caption, outline score, split), sprites/captions.json
- Sources: DCSS 32x32 tiles via ocdbytes/pixl-data 410 | bluecarrot16 LPC food 158 | Lazarini 496 RPG icons (OGA CC0, 34px cropped to 32) 139 |
  Idylwild's Inventory 50 | TheKingPhoenix food 25 + tools 7 | Jetrel food 15 | Kenney Pixel Shmup 10 | yd tools 4 | ArlanTR food 4 |
  Scrittl spaceships 4 | Buch platformer 4 | wender sports cars 4 | Fleurman tools 3 | 1 each: OGA food-other, Sunny Land, Jull spaceship.
- Filters: binary alpha, semi-transparent black drop shadows removed, 40-940 opaque px, <=2 near-magenta px, >=3 colours,
  near-duplicates removed (pHash<=5 and mean pixel diff<14), at most 1 per DCSS variant group / 1 per LPC stack group.
- Captions: Gemini (gemini-flash-latest) looked at each sprite with its filename hint and wrote a 2-6 word everyday caption.
- Sheets (nearest-neighbour x8, 256 px cells, #FF00FF background; "dark outline" only when every cell has a dark outline (>=60% of edge px lum<70)):
  - dataset/sheets_2x2_512/{train 201, eval 11}  (each train sprite once)
  - dataset/sheets_4x4_1024/{train 153, eval 3}  (3 shuffles of the train sprites)
  - each NNNN.png has NNNN.txt; manifest.jsonl per folder
- Preview: preview_64_train_sprites.png (+ .txt with captions)
- musubi caches built on the box CPU (musubi-tuner f8a1b03): cache/2x2_512 (773 MB), cache/4x4_1024 (1.3 GB)

## Training (musubi-tuner, Krea 2 Raw ConvRot int8, on the 3080 Ti) - NOT STARTED
See TRAIN_COMMANDS.sh
