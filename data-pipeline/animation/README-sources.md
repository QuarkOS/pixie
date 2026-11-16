# Collect artist-made animation sequences

The first-round goal is approximately **500 curated four-frame artist
sequences**. The operator measured 126 imported sequences from five packs,
and 195 built sheets, of which only 35 were artist-made. These totals describe
different stages. Relative to 35 artist sheets, the remaining goal is about
465, before further review or duplicate removal. Procedural and generated
examples do not fill that gap.

Public pages were checked for advertised material; archives were not downloaded
or inspected for this change. **Every expected sequence count below is an
untested planning estimate, not a measured yield, source inventory or acceptance
guarantee.** Even the low end can become zero after file inspection, holdout
exclusion and review. Links support the material descriptions; estimated yields
are our inference. License fields remain optional metadata for this project.
Every asset still needs a public source URL and content hash.

## Count useful motion, then replace estimates with actual yields

Count one reviewed subject/action/direction sequence with useful motion and
four selected chronological phases. Inspect the complete original action to
select phases and durations. Preserve complete source masters; taking the first
four frames of a longer action does not establish a valid loop. The builder
requires explicitly curated four-frame inputs and never selects or pads frames.

Recolors, mirrored directions, repeated windows of one action, atlas and
individual-file duplicates, clothing combinations and repackaged copies do
not increase the unique-motion count. Assign related variants the same lineage
across packs. Keep attacks and deaths non-looping when appropriate. Do not
duplicate poses to turn a shorter action into an apparent four-phase example.

For every pack, record archive SHA256, per-file hashes, imported/accepted counts,
source kind, rejection reasons, unique motion count, train/eval counts and
holdout exclusions. Replace the estimates in a local collection ledger. Track
subject and action coverage too: many actions on one character are not broad
subject coverage. Start with action-rich timed packs, then fill gaps in animals,
flying, reactions, attacks and objects rather than cosmetic variations.

## Free itch.io packs and free samplers

Expected counts refer to candidate four-phase actions after obvious duplicate
variants are removed. All need review; paid extras are outside this queue.

| Public source page | Expected sequences (estimate) | Selection and preparation |
|---|---:|---|
| [Pixel Adventure 1](https://pixelfrog-assets.itch.io/pixel-adventure-1) | 16–28 | Character/enemy actions and useful effects; timed PNG animations. Exclude static tiles. |
| [Kings and Pigs](https://pixelfrog-assets.itch.io/kings-and-pigs) | 24–40 | Player, pigs and effects. Export editable Aseprite files as JSON plus PNG, or map PNG strips explicitly. |
| [Treasure Hunters](https://pixelfrog-assets.itch.io/treasure-hunters) | 18–30 | Pirate player, enemies and moving traps. The page documents animation timing. Exclude scenery and UI. |
| [Forest 16x16 platformer assets](https://anokolisa.itch.io/sidescroller-pixelart-sprites-asset-pack-forest-16x16) | 12–24 | Warrior, boar, snail and flying enemy actions. Check motion survives normalization. |
| [Animated Pixel Hero](https://rvros.itch.io/animated-pixel-hero) | 12–24 | Broad player action set on one identity. Do not confuse action count with subject coverage. |
| [Fantasy Knight](https://aamatniekss.itch.io/fantasy-knight-free-pixelart-animated-character) | 8–14 | Combat and locomotion with large padded cells: use a shared crop or separate 64x64 collection. |
| [Warrior Free Animation Set](https://clembod.itch.io/warrior-free-animation-set) | 8–14 | Run, combat, hit, death and other actions. Skip actions with too few useful phases. |
| [Character Animation Asset Pack](https://muchopixels.itch.io/character-animation-asset-pack) | 8–16 | Character motion references; inspect actual free action inventory and order before collection. |
| [Free Pixel Character](https://fatberry.itch.io/free-pixel-character) | 4–7 | Compact player action set with folders/sheets. Declare order and timing explicitly. |
| [Dungeon Tileset II](https://0x72.itch.io/dungeontileset-ii) | 12–24 | Small characters/enemies. Static tiles and short pose cycles do not qualify automatically. |
| [Free Pixel Art Characters sampler](https://torpilstudio.itch.io/free-pixel-art-characters) | 30–48 | Timed strips plus JSON; a repack of existing artwork. Trace upstream credits and group matching identities. Do not count its paid bundle inventory. |
| [Sci-Fi Character Pack 12 free sprite](https://penusbmic.itch.io/sci-fi-character-pack-12) | 4–8 | Only the free character, not the paid bundle. Adds mechanical subjects; inspect actual free actions. |

The Pixel Frog pages describe CC0 releases; other sources have their own terms.
Pixel Adventure 2 is not assumed free and is outside this queue. Use the manual
provenance path below when itch.io supplies a temporary signed link.

## OpenGameArt character and small-creature packs

| Public source page | Expected sequences (estimate) | Selection and preparation |
|---|---:|---|
| [Simple Character Template](https://opengameart.org/content/simple-character-template) | 2–6 | Idle/walk GIFs and sheet. GIF metadata supplies timing; sheet cells need a map. |
| [Animated Birds 32x32](https://opengameart.org/content/animated-birds-32x32) | 4–12 | Flying subjects. For exact magenta backgrounds use `--key-rgb 255 0 255`, then review edges. |
| [Animated Warrior](https://opengameart.org/content/animated-warrior) | 6–10 | Character locomotion, gestures and combat. Group body variants. |
| [Animated Wizard](https://opengameart.org/content/animated-wizard) | 6–10 | Spellcaster actions; group related character versions across splits. |
| [Animated Cleric](https://opengameart.org/content/animated-cleric) | 6–10 | Character actions; distinguish gestures from attacks in captions. |
| [Animated Goblins](https://opengameart.org/content/animated-goblins) | 6–10 | Knife and hammer subjects. Keep related weapon/body variants grouped. |
| [Animated Orcs](https://opengameart.org/content/animated-orcs) | 6–10 | Spear fighter and hooded character actions. Group related variants. |
| [Animated Skeleton](https://opengameart.org/content/animated-skeleton) | 3–5 | Undead actions. Check that four-phase death/gesture selections remain interpretable. |
| [Four-direction Animated Warrior](https://opengameart.org/content/4-direction-animated-warrior) | 8–13 | Labelled sheet and readme. Count distinct directional motion; weapon overlays remain variants. |

These related packs share an art family. They improve action coverage but must
not dominate evaluation or replace animal/object coverage. Public availability
alone does not mean CC0.

## LPC actions, animals and larger source cells

Use the [LPC collection](https://opengameart.org/content/lpc-collection) and
[creature index](https://opengameart.org/content/lpc-creatures-and-animals) for
discovery, not as additional packs to count. Prefer one current upstream version
and retain lineage links across shared bases, revisions and recolors.

| Public source page | Expected sequences (estimate) | Selection and preparation |
|---|---:|---|
| [LPC Character Bases](https://opengameart.org/content/lpc-character-bases) | 36–72 | Current body/action/direction families; exclude head, clothing and skin-color combinatorial inflation. Group shared bases. |
| [LPC Goblin Full Sheet](https://opengameart.org/content/lpc-goblin-full-sheet) | 12–20 | Expanded actions; inspect alignment and do not also count the older partial pack. |
| [LPC Wolf Animation](https://opengameart.org/content/lpc-wolf-animation) | 10–16 | Walk, run, bite, howl, death. Recolored sheets share motion and lineage. |
| [LPC Horses](https://opengameart.org/content/lpc-horses) | 8–14 | Walk, eat, gallop; mirrored directions and recolors are not additional unique motions. |
| [LPC Wild Boar](https://opengameart.org/content/lpc-wild-boar) | 6–10 | Walk, attack, death with armor variants. Review attack alignment; do not silently shift individual frames to hide motion. |
| [LPC Bears, Deer, Lions and More](https://opengameart.org/content/lpc-bears-deer-lions-and-more) | 24–40 | Walk, attack, death, eating and swimming. Dedup combined/individual files and colors. Use shared crops or larger canvases. |
| [LPC Birds](https://opengameart.org/content/lpc-birds) | 12–20 | Flying/walking directions. Group matching silhouettes, recolors and other bird-pack copies. |
| [LPC Fish](https://opengameart.org/content/lpc-fish) | 4–6 | Swimming small/regular fish; underwater tint is a variant. Verify useful source phase count. |
| [LPC Push and Carry](https://opengameart.org/content/lpc-character-animations-push-and-carry) | 4–8 | Adds action coverage on a shared base. Exclude static idle; join the base lineage used elsewhere. |
| [LPC Rat, Cat and Dog](https://opengameart.org/content/lpc-rat-cat-and-dog) | 0 for this round | Page describes short walk cycles with fewer than four poses. Reserve for a variable-length format; do not pad to meet this target. |

These ranges are a backlog, not an additive promise of reaching 500. Cross-pack
duplicates and failed curation reduce the total. Inspect measured per-pack
yield before expanding further; if needed, add distinct LPC body/action/direction
families rather than counting cosmetic edits as new motion.

## Kenney pose tiles with explicit timing

| Public source page | Expected sequences (estimate) | Selection and preparation |
|---|---:|---|
| [Pixel Platformer](https://kenney.nl/assets/pixel-platformer) | 0–4 | Untimed pixel-art poses; page does not establish complete actions. Inspect pose families and declare order. |
| [Platformer Characters](https://kenney.nl/assets/platformer-characters) | 0–12 | Untimed poses, not necessarily native pixel art. Review coherent motion and style; expressions/cosmetic variants do not count. |
| [Mini Dungeon](https://kenney.nl/assets/mini-dungeon) | 0 direct imports | Animated 3D assets; needs a separate fixed-camera rendering pipeline before PNG/GIF import. |
| [Mini Arena](https://kenney.nl/assets/mini-arena) | 0 direct imports | Animated 3D assets, not an existing pixel-animation sheet corpus. |

Kenney's public packs are CC0, but an asset count is not an animation count.
For inspected untimed poses, declare `frames` or sheet `order` and supply
`--default-duration-ms 100`. This illustrative playback duration is an annotation,
not recovered timing. Source GIF/Aseprite timings and explicit frame/sequence
durations retain priority. Ingest records the fallback and `timing_origin`;
it does not infer order or admit static pose collections as motion.

## Preserve provenance without signed URLs

See [download provenance](README-download.md) for either acquisition route.
For itch.io, use the browser's free-download flow, retain the archive locally,
then run from `data-pipeline`:

```sh
$PY -m animation.download --local-file "$DATA_ROOT/downloads/pack.zip" --source-url "https://pixelfrog-assets.itch.io/kings-and-pigs" --output "$DATA_ROOT/animation-downloads/kings-and-pigs-v1"
```

The new folder contains byte-identical `asset.bin` and `download.json` with the
public source page, SHA256, byte count and `manual_download` acquisition. It
does not record the private local path or signed URL. Direct permanent public
URLs still use `--url`. Neither route extracts archives or automates accounts.

The downloader hashes the downloaded archive. Ingest separately hashes the
selected source files and preserves their original bytes. Retain the archive
and its download record alongside the imported collection so the per-file
records can be traced back to the downloaded pack. A hash identifies the
actual bytes received; this source list does not pin future pack revisions.

Before training sheets, apply the repository's normalized Blind-50/Eval-50
policy, review every sequence and audit lineage-grouped splits. Do not
regenerate, reformat or edit either frozen holdout file.
