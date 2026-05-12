"""
download_images.py
------------------
Downloads Pokémon sprites for every entry in pokedex_clustered.csv.

Primary:  https://img.pokemondb.net/artwork/vector/{slug}.png  (high-res artwork)
Fallback: https://img.pokemondb.net/sprites/home/normal/{slug}.png (HOME sprites)

Regional/alternate forms that lack official artwork fall back to HOME sprites
automatically — no manual intervention needed.

Run once:  python download_images.py
Re-run to retry failures after patching MANUAL_SLUGS.
"""

import re
import time
import requests
import pandas as pd
from pathlib import Path

HERE     = Path(__file__).parent
IMG_DIR  = HERE / "images"
IMG_DIR.mkdir(exist_ok=True)

ARTWORK_URL = "https://img.pokemondb.net/artwork/vector/{}.png"
SPRITE_URL  = "https://img.pokemondb.net/sprites/home/normal/{}.png"
HEADERS     = {"User-Agent": "Mozilla/5.0 (research/portfolio project)"}

# ── manual slug overrides ─────────────────────────────────────────────────────
# Only needed when the algorithm can't derive the correct slug.
# Format: csv_name (exact) → PokémonDB slug
MANUAL_SLUGS = {
    # Rotom — form name repeats the base name, algorithm over-includes it
    "Rotom  Fan Rotom":             "rotom-fan",
    "Rotom  Frost Rotom":           "rotom-frost",
    "Rotom  Heat Rotom":            "rotom-heat",
    "Rotom  Mow Rotom":             "rotom-mow",
    "Rotom  Wash Rotom":            "rotom-wash",
    # Kyurem / Necrozma — base name embedded in form name
    "Kyurem  Black Kyurem":         "kyurem-black",
    "Kyurem  White Kyurem":         "kyurem-white",
    "Necrozma  Dawn Wings Necrozma":"necrozma-dawn-wings",
    "Necrozma  Dusk Mane Necrozma": "necrozma-dusk-mane",
    # Hoopa — "Confined" is the default; maps to base slug
    "Hoopa  Hoopa Confined":        "hoopa",
    "Hoopa  Hoopa Unbound":         "hoopa-unbound",
    # Default/hero forms that share the base slug
    "Keldeo  Ordinary Form":        "keldeo",
    "Darmanitan  Standard Mode":    "darmanitan",
    # Galarian Standard Mode: region detection fires on "Galarian" and picks
    # "Standard Mode" as the base name — override needed
    "Darmanitan  Galarian Standard Mode": "darmanitan-galarian-standard",
    # Zacian / Zamazenta
    "Zacian  Hero of Many Battles": "zacian",
    "Zamazenta  Hero of Many Battles": "zamazenta",
    "Zacian  Crowned Sword":        "zacian-crowned",
    "Zamazenta  Crowned Shield":    "zamazenta-crowned",
    # Urshifu — single strike is the default
    "Urshifu  Single Strike Style": "urshifu",
    "Urshifu  Rapid Strike Style":  "urshifu-rapid-strike",
    # Tauros — combat breed uses a different slug format than aqua/blaze
    "Tauros  Combat Breed":         "tauros-paldean-combat",
    # Maushold — family-of-three sprite uses same slug as default
    "Maushold  Family of Three":    "maushold",
    # Oricorio Pa'u — apostrophe edge case (algorithm handles it, but explicit is safer)
    "Oricorio  Pa'u Style":         "oricorio-pau",
}

REGION_MAP = {
    "Alolan":  "alolan",
    "Galarian": "galarian",
    "Hisuian": "hisuian",
    "Paldean": "paldean",
}

# Trailing words stripped from form names before slugifying
STRIP_SUFFIXES = [
    " Forme", " Form", " Mode", " Style", " Size",
    " Cloak", " Breed", " Plumage", " Mask",
]


def simple_slug(name: str) -> str:
    s = name.lower().strip()
    s = s.replace("♀", "-f").replace("♂", "-m")
    s = s.replace("'", "").replace(".", "").replace(":", "").replace("%", "")
    s = s.replace("é", "e")
    s = re.sub(r"[^a-z0-9\-]", "-", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("-")


def name_to_slug(csv_name: str) -> str:
    if csv_name in MANUAL_SLUGS:
        return MANUAL_SLUGS[csv_name]

    if "  " not in csv_name:
        return simple_slug(csv_name)

    base, form = csv_name.split("  ", 1)
    base, form = base.strip(), form.strip()

    form_words = form.split()
    if form_words[0] in REGION_MAP:
        region    = REGION_MAP[form_words[0]]
        base_name = " ".join(form_words[1:])
        return simple_slug(base_name) + "-" + region

    form_stripped = form
    for suffix in STRIP_SUFFIXES:
        if form_stripped.endswith(suffix):
            form_stripped = form_stripped[: -len(suffix)].strip()
            break

    base_first = base.split()[0]
    if form_stripped.startswith(base_first + " "):
        form_stripped = form_stripped[len(base_first):].strip()
    elif form_stripped == base_first:
        form_stripped = ""

    suffix_slug = simple_slug(form_stripped) if form_stripped else ""
    base_slug   = simple_slug(base)
    return f"{base_slug}-{suffix_slug}" if suffix_slug else base_slug


def display_name(csv_name: str) -> str:
    if "  " in csv_name:
        base, form = csv_name.split("  ", 1)
        return f"{base.strip()} ({form.strip()})"
    return csv_name


def try_download(slug: str) -> tuple[bytes | None, str]:
    """Try artwork first, fall back to HOME sprite. Returns (data, source_url)."""
    for url_tmpl in [ARTWORK_URL, SPRITE_URL]:
        url = url_tmpl.format(slug)
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            if r.status_code == 200:
                return r.content, url
        except Exception:
            pass
        time.sleep(0.05)
    return None, ""


# ── main ──────────────────────────────────────────────────────────────────────
df      = pd.read_csv(HERE / "pokedex_clustered.csv")
results = []

print(f"Downloading images for {len(df)} Pokémon …\n")

for _, row in df.iterrows():
    csv_name = row["Name"]
    slug     = name_to_slug(csv_name)
    display  = display_name(csv_name)
    out_path = IMG_DIR / f"{slug}.png"

    if out_path.exists():
        print(f"  [skip]  {display}")
        results.append(dict(csv_name=csv_name, display_name=display,
                            slug=slug, path=str(out_path), status="cached"))
        continue

    data, src = try_download(slug)
    if data:
        out_path.write_bytes(data)
        source = "art" if "artwork" in src else "sprite"
        print(f"  [ok/{source}]  {display}  ->  {slug}")
        results.append(dict(csv_name=csv_name, display_name=display,
                            slug=slug, path=str(out_path), status="ok"))
    else:
        print(f"  [FAIL]  {display}  ->  {slug}")
        results.append(dict(csv_name=csv_name, display_name=display,
                            slug=slug, path="", status="fail"))

    time.sleep(0.1)

# ── summary ───────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv(HERE / "image_map.csv", index=False)

ok  = results_df["status"].isin(["ok", "cached"]).sum()
print(f"\nDone: {ok}/{len(results_df)} images saved.")

failed = results_df[~results_df["status"].isin(["ok", "cached"])]
if len(failed):
    print(f"\n{len(failed)} failures — add to MANUAL_SLUGS and re-run:")
    for _, r in failed.iterrows():
        print(f"  '{r['csv_name']}':  '{r['slug']}',")
