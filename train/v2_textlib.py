"""Shared text rules for Pixel AI v2 (used by the cache build, the trainer's
sample prompts, and later inference). Pure Python + the HF `tokenizers` file.

Caption -> clean_caption() -> CLIP BPE (openai/clip-vit-base-patch32
tokenizer.json, lowercase, no BOS/EOS) -> remap to the v2 vocabulary
(0 = PAD, 1 = UNK, 2.. = CLIP ids seen in the v2 training captions)
-> truncate / pad to TEXT_LEN.
"""
import re

TEXT_LEN = 24
PAD_ID = 0
UNK_ID = 1

# Unambiguous franchise / trademark names only. Generic English words that
# also happen to be names (sonic, tails, knuckles, halo, steam, creeper,
# toad, link, pac, steve) are deliberately NOT listed.
TRADEMARKS = [
    "binding of isaac", "hatsune miku", "pokemon", "pokémon", "pikachu", "pokeball", "poke ball",
    "super mario", "mario", "luigi", "bowser", "yoshi", "zelda", "triforce", "hyrule",
    "sonic the hedgehog", "minecraft", "enderman", "miku", "isaac", "kirby", "metroid", "samus",
    "mega man", "megaman", "pac-man", "pacman", "pac man", "tetris", "star wars", "darth vader",
    "yoda", "stormtrooper", "mickey mouse", "disney", "marvel", "spider-man", "spiderman",
    "batman", "superman", "nintendo", "playstation", "xbox", "fortnite", "among us", "amogus",
    "overwatch", "warcraft", "starcraft", "skyrim", "final fantasy", "chocobo", "moogle",
    "undertale", "terraria", "stardew", "hello kitty", "sanrio", "dragon ball", "goku",
    "naruto", "lego", "coca-cola", "coca cola", "pepsi", "mcdonald's", "mcdonalds",
    "harry potter", "hogwarts", "gandalf", "frodo", "shrek", "spongebob", "simpsons",
    "donkey kong", "street fighter", "mortal kombat", "space invaders", "galaga", "genshin",
    "pikmin", "kingdom hearts", "master chief", "pip-boy", "pipboy", "fallout", "witcher",
    "dark souls", "elden ring", "hollow knight", "cuphead", "shovel knight", "animal crossing",
    "splatoon", "smash bros", "godzilla", "totoro", "ghibli", "sailor moon", "barbie",
    "hot wheels", "iphone", "tardis", "dalek", "star trek", "ghostbusters",
]
_TM = re.compile(r"\b(" + "|".join(re.escape(t) for t in sorted(TRADEMARKS, key=len, reverse=True)) + r")\b", re.I)
_WS = re.compile(r"\s+")
_DANGLING = re.compile(r"\b(from|of|in the style of|style|like)\s*$", re.I)

# Horizontal flip is skipped when the caption carries direction or glyphs.
NOFLIP = re.compile(
    r"\b(left|right|east|west|letters?|lettering|numbers?|numerals?|digits?|text|texts|"
    r"arrows?|clocks?|clockwise|counterclockwise|anticlockwise|signs?|signpost|signposts|signboard)\b",
    re.I,
)


def clean_caption(caption: str) -> tuple[str, bool]:
    """Strip trademark names. Returns (clean, changed)."""
    out = _TM.sub(" ", caption)
    changed = out != caption
    if changed:
        out = _WS.sub(" ", out).strip(" ,-")
        out = _DANGLING.sub("", out).strip(" ,-")
    return out, changed


def allow_flip(caption: str) -> bool:
    return NOFLIP.search(caption) is None


class V2Tokenizer:
    def __init__(self, tokenizer_json: str, remap: dict[int, int]):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(tokenizer_json)
        self.remap = remap

    def clip_ids(self, captions: list[str]) -> list[list[int]]:
        return [e.ids for e in self.tok.encode_batch(captions, add_special_tokens=False)]

    def encode(self, captions: list[str]) -> list[list[int]]:
        rows = []
        for ids in self.clip_ids(captions):
            v = [self.remap.get(i, UNK_ID) for i in ids][:TEXT_LEN]
            rows.append(v + [PAD_ID] * (TEXT_LEN - len(v)))
        return rows
