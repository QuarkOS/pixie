# Pixie staged API (login + cond + share)

These endpoints are on a local app (`127.0.0.1:8791` by default). They are not a live deployment. Existing A/B, vote and label routes are unchanged. Votes in `labels.jsonl` / `chat.jsonl` are never rewritten; new chat votes may gain an optional `thumb` path and a `cond` field.

Public base for share URLs is `AUTH_PUBLIC_URL` (on staging: `http://127.0.0.1:8791`; at go-live: `https://pixie.example.invalid`).

## Palette and style (s7 cond)

The live v3-full-s7 model already has 3 palette slots + 1 style slot. Plain chat today sends the default: palette ids `4097,4097,4097` (`NO_PALETTE`) and style `[0,0,0]` (ANY fill / outline / colour-count).

- Palette ids are indexes into the 4097-colour table (1–4096; 0 is transparent and is not a valid hint; 4097 = empty slot). Up to 3 ids. Named palettes (ember, gold, forest, …) are convenience sets of 3 ids.
- Style: `fill` 0–3 (any / small &lt;80 opaque px / medium 80–180 / full &gt;180), `outline` 0–2 (any / dark / soft), `colours` 0–3 (any / ≤12 / 13–24 / &gt;24). Training used real tags on 40% of samples; outline is 99.8% dark in the train set, so that tag is weak.

### `GET /api/chat/info`

Same queue snapshot as before (`waiting`, `running`, `avg_ms`, `queue_max`, `worker_online`, `prompt_max`) plus:

```json
{ "cond": { "palettes": [{ "id": "auto"|"gold"|..., "label": "...", "ids": [..]|"null", "swatches": [{ "hex": "#d4a33e", "id": 2343 }] }], "style": { "fill": [...], "outline": [...], "colours": [...] }, "model": { "n_pal": 3, "style": true, "no_palette": 4097, "text_len": 18 } } }
```

### `POST /api/chat/submit`

Body, in addition to `{ token, prompt }`:

- `palette`: `"auto"` | a named id | `[id, id?, id?]`
- `style`: `{ fill, outline, colours }` or `[fill, outline, colours]` (names or ints)

Omitted fields default to auto/any. Invalid values return **400**. The worker job payload includes `cond: { palette: [..], style: [f,o,c] }` only when not default, so an old worker can ignore it.

`GET/POST /api/chat/status` returns `cond` on the job view when a non-default hint was used.

## Download

### `POST /api/chat/download`

Body: `{ token, job_id, index, size?, bg? }`

- `index`: 0–3 for the shown top-4, or a candidate `idx` from grid mode
- `size`: 16, 32, 64, 128, 256, 512 (default), or 1024
- `bg`: `transparent` (default) | `black` | `white` | `surface` (#111) | `#rrggbb`

Response: `image/png` (attachment). The job must still be in memory (about one hour after it finishes). For a durable link, create a share.

## Share links

`/s/<id>` is the public page. It does **not** collide with A/B sprite files, which stay at `/s/<12-hex>.png` (static files win if they exist). Share ids are 14-character base64url.

### `POST /api/share`

Body: `{ token, job_id, index }` → `{ id, url, path, prompt, ts }`

Stores a 16×16 PNG and a 1200×630 OG image under the staged data dir (`DATA_DIR/shares/`). Does not write votes.

### `GET /api/share/:id`

JSON metadata and image paths.

### `GET /s/:id`

HTML page (hooks: `#share`, `#shareSprite`, `#sharePrompt`, `#shareTools`, `#shareDownload16`, `#shareDownload512`, `#shareDownload512t`, `#shareCopy`). Look is left to the UI designer. OG/Twitter tags point at `/s/:id/og.png`.

### `GET /s/:id/png?size=16|512&bg=transparent|black|white|surface|#rrggbb`

Native or nearest-neighbour upscaled PNG.

### `GET /s/:id/og.png`

1200×630 preview (sprite centred on a dark card, prompt caption) for X/Discord.

## Admin thumbnail (new votes only)

New chat votes may include `thumb: "/api/admin/chat_thumb?job=<job_id>"`. The 64×64 PNG is stored under `DATA_DIR/chat-thumbs/`. Old votes are not rewritten and will still have no thumbnail. The admin table already looks at `c.thumb`.

`GET /api/admin/chat_thumb?job=...` requires an admin session.

## Markup hooks for the UI designer (do not rename)

Chat (`public/chat.html`, `public/chat.js`):

- `#condBar`, `#palettePick`, `#paletteSwatches`, `#styleFill`, `#styleOutline`, `#styleColours`
- `.px-sprite-wrap`, `.px-sprite-tools`, `[data-action=download]`, `[data-action=share]`

Share: see `public/share.html` placeholders `__TITLE__` etc. (server-rendered; keep the `__…__` tokens if you restyle the template).
