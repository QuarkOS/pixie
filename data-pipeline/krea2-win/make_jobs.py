#!/usr/bin/env python3
import argparse, json, math
SUBJECTS = """red apple|banana|orange carrot|slice of bread|cup of coffee|glass of milk|cheeseburger|pepperoni pizza|
fried egg|chocolate donut|ice cream cone|wedge of cheese|green pear|bunch of grapes|strawberry|watermelon slice|
house key|padlock|wooden chair|table lamp|alarm clock|smartphone|laptop|television|
toothbrush|bar of soap|bath towel|umbrella|backpack|sneaker|baseball cap|t-shirt|
eyeglasses|wristwatch|wallet|coin purse|pencil|scissors|open book|envelope|
potted plant|sunflower|green tree|cactus in a pot|orange cat|brown dog|goldfish|yellow duck|
red bicycle|blue car|yellow school bus|paper airplane|soccer ball|basketball|guitar|headphones|
hammer|screwdriver|paint brush|watering can|cooking pot|frying pan|teapot|kitchen knife|
candle|light bulb|battery|camera|gift box|red balloon|teddy bear|rubber duck""".replace("\n", "").split("|")
SUBJECTS = [s.strip() for s in SUBJECTS if s.strip()]
assert len(SUBJECTS) == 72, len(SUBJECTS)

def sigmas(n, s=math.exp(1.15)):
    out = []
    for k in range(n + 1):
        t = 1 - k / n
        out.append(round(s * t / (1 + (s - 1) * t), 8))
    return out

def sheet_prompt(items, g, px):
    head = f"pixel art {px}x{px} game icons, {g}x{g} sprite sheet, "
    tail = ", dark outline, plain magenta background"
    if g == 2:
        tl, tr, bl, br = items
        return head + f"top left: {tl}, top right: {tr}, bottom left: {bl}, bottom right: {br}" + tail
    if g == 3:
        rows = ["top row", "middle row", "bottom row"]
    else:
        rows = [f"row {r+1}" for r in range(g)]
    parts = [f"{rows[r]}: " + ", ".join(items[r*g:(r+1)*g]) for r in range(g)]
    return head + ", ".join(parts) + tail

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=2); ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=5); ap.add_argument("--lora", type=float, default=1.0)
    ap.add_argument("--px", type=int, default=16); ap.add_argument("--sheets", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1000); ap.add_argument("--outdir", required=True)
    ap.add_argument("--warmup", type=int, default=1); ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--shift", type=float, default=1.15)
    ap.add_argument("--vae-tile", type=int, default=0)
    ap.add_argument("-o", required=True)
    a = ap.parse_args()
    per = a.grid * a.grid
    sig = sigmas(a.steps, math.exp(a.shift))
    if a.steps == 4 and a.shift == 1.15:
        sig = [1.0, 0.90453077, 0.75951091, 0.51284410, 0.0]  # review plan, exact
    jobs = []
    for k in range(-a.warmup, a.sheets):
        if k < 0:
            items = ["iron sword", "wooden shield", "red heart", "gold coin", "magic potion", "treasure chest", "bow", "arrow", "skull", "crown", "torch", "scroll", "gem", "bomb", "boot", "ring"][:per]
            out = f"{a.outdir}/warmup.png"; seed = 7
        else:
            base = (a.offset + k * per) % len(SUBJECTS)
            items = [SUBJECTS[(base + i) % len(SUBJECTS)] for i in range(per)]
            out = f"{a.outdir}/sheet-{k:03d}.png"; seed = a.seed + k
        jobs.append({"prompt": sheet_prompt(items, a.grid, a.px), "items": items, "grid": a.grid, "seed": seed,
                     "width": a.size, "height": a.size, "steps": a.steps, "sigmas": sig,
                     "lora_mult": a.lora, "out": out, "vae_tile": a.vae_tile})
    json.dump(jobs, open(a.o, "w"), indent=1)

if __name__ == "__main__":
    main()
