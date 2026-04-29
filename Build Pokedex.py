"""
build_pokedex.py
----------------
Python translation of the R data wrangling pipeline.
Scrapes PokemonDB and Bulbapedia, merges to a fully-evolved Pokedex,
adds Generation, and removes invalid / multi-form entries.

Requirements:
    pip install pandas requests lxml beautifulsoup4
"""

import re
import pandas as pd
import requests
import io

# ---------------------------------------------------------------------------
# 1. Scrape source tables
# ---------------------------------------------------------------------------

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research project)"}

URL1 = "https://pokemondb.net/pokedex/all"
URL2 = "https://bulbapedia.bulbagarden.net/wiki/List_of_fully_evolved_Pok%C3%A9mon_by_base_stats"

print("Fetching PokemonDB...")
r1 = requests.get(URL1, headers=HEADERS, timeout=15)
r1.raise_for_status()
dex = pd.read_html(io.StringIO(r1.text))[0].copy()

print("Fetching Bulbapedia...")
r2 = requests.get(URL2, headers=HEADERS, timeout=15)
r2.raise_for_status()
fe_dex = pd.read_html(io.StringIO(r2.text))[0].copy()

print(f"  PokemonDB rows  : {len(dex)}")
print(f"  Bulbapedia rows : {len(fe_dex)}")

# ---------------------------------------------------------------------------
# 2. Rename columns to match R output
#    PokemonDB  : #, Name, Type, Total, HP, Attack, Defense, Sp. Atk, Sp. Def, Speed
#    Bulbapedia : #, Name, Type, HP, Attack, Defense, Sp. Atk, Sp. Def, Speed, Total, ...
# ---------------------------------------------------------------------------

# Rename the dex-number column (first col) to Dex_Number
dex.rename(columns={dex.columns[0]: "Dex_Number"}, inplace=True)
fe_dex.rename(columns={fe_dex.columns[0]: "FE_Dex_Number"}, inplace=True)

# PokemonDB uses the name in column 1
dex_name_col   = dex.columns[1]    # "Name"
fe_name_col    = fe_dex.columns[1] # "Pokémon" or "Name"

# ---------------------------------------------------------------------------
# 3. Filter dex to only fully-evolved Pokemon  (R: dex[dex[,1] %in% fe_dex[,1],])
#    Match on Dex Number (most reliable, avoids name encoding issues)
# ---------------------------------------------------------------------------

fe_numbers = set(fe_dex["FE_Dex_Number"].dropna().astype(int))
pokedex = dex[dex["Dex_Number"].isin(fe_numbers)].copy()
print(f"  After FE filter : {len(pokedex)} rows")

# ---------------------------------------------------------------------------
# 4. Add Generation column
# ---------------------------------------------------------------------------

def assign_generation(n):
    if   1   <= n <= 151: return 1
    elif 152 <= n <= 251: return 2
    elif 252 <= n <= 386: return 3
    elif 387 <= n <= 493: return 4
    elif 494 <= n <= 649: return 5
    elif 650 <= n <= 721: return 6
    elif 722 <= n <= 809: return 7
    elif 810 <= n <= 905: return 8
    elif n   >= 906:      return 9
    else: return None

pokedex["Generation"] = pokedex["Dex_Number"].apply(assign_generation)

# ---------------------------------------------------------------------------
# 5. Remove invalid / multi-form entries
# ---------------------------------------------------------------------------

INVALID = [
    "Mega ", "Partner", "Resolute Form", "Complete Forme",
    "Ash-Greninja", "Minior", "Aegislash", "Wishiwashi",
    "Pirouette Form", "Sunny Form",
    "Rainy Form", "Snowy Form",
    "TerapogosNormal Form", "Stellar Form",
    "Droopy Form", "Stretchy Form",
    "Primal",
    "Ultra Necrozma", "Zen Mode",
    "White-Striped Form", "Blue-Striped Form",
    "Family of Four",
    "Low Key Form", "Palafin", "Blue Plumage",
    "Yellow Plumage", "White Plumage",
    "Morpeko", "Eiscue", "MeowsticMale",
    "Three-Segment Form", "Eternamax",
    "Terapagos  Normal Form"
]

# Build a single regex pattern and filter on the Name column
invalid_pattern = "|".join(re.escape(s) for s in INVALID)
mask = pokedex[dex_name_col].str.contains(invalid_pattern, na=False)
removed = pokedex[mask][dex_name_col].tolist()
pokedex = pokedex[~mask].reset_index(drop=True)

print(f"  Removed {len(removed)} invalid rows: {removed[:10]}{'...' if len(removed) > 10 else ''}")
print(f"  Final pokedex   : {len(pokedex)} fully-evolved Pokemon")

# ---------------------------------------------------------------------------
# 6. Standardise stat column names for downstream ML
# ---------------------------------------------------------------------------

rename_map = {
    "Sp. Atk": "Sp_Atk",
    "Sp. Def":  "Sp_Def",
    "Attack":   "Attack",
    "Defense":  "Defense",
    "Speed":    "Speed",
    "HP":       "HP",
    "Total":    "Total",
}
pokedex.rename(columns=rename_map, inplace=True)

STAT_COLS = ["HP", "Attack", "Defense", "Sp_Atk", "Sp_Def", "Speed"]

# ---------------------------------------------------------------------------
# 7. Save
# ---------------------------------------------------------------------------

out_path = "pokedex_fe.csv"
pokedex.to_csv(out_path, index=False)
print(f"\nSaved → {out_path}")
print(pokedex[["Dex_Number", dex_name_col, "Type", "Generation"] + STAT_COLS].head(10).to_string(index=False))