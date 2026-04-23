# src/generate_future_dataset.py

import pandas as pd
import numpy as np

# ==================================================
# CONFIG
# ==================================================
INPUT_CSV  = "Datasets/areas_10k_tumor_only.csv"
OUTPUT_CSV = "Datasets/future_growth_dataset.csv"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ==================================================
# LOAD CURRENT AREAS
# ==================================================
df = pd.read_csv(INPUT_CSV)

# Use area in cm² for cleaner scale
current = df["area_cm2"].values

print("Tumor samples loaded:", len(current))

# ==================================================
# FUTURE SIMULATION FUNCTION
# ==================================================
def simulate_future(area):
    """
    Generate realistic future tumor sizes:
    30d, 60d, 90d
    """

    scenario = np.random.choice(
        ["slow", "moderate", "aggressive", "stable", "shrink"],
        p=[0.30, 0.25, 0.15, 0.15, 0.15]
    )

    if scenario == "slow":
        r1 = np.random.uniform(1.02, 1.08)
        r2 = np.random.uniform(1.02, 1.08)
        r3 = np.random.uniform(1.02, 1.08)

    elif scenario == "moderate":
        r1 = np.random.uniform(1.08, 1.18)
        r2 = np.random.uniform(1.08, 1.18)
        r3 = np.random.uniform(1.08, 1.18)

    elif scenario == "aggressive":
        r1 = np.random.uniform(1.18, 1.35)
        r2 = np.random.uniform(1.18, 1.35)
        r3 = np.random.uniform(1.18, 1.35)

    elif scenario == "stable":
        r1 = np.random.uniform(0.98, 1.02)
        r2 = np.random.uniform(0.98, 1.02)
        r3 = np.random.uniform(0.98, 1.02)

    else:  # shrink
        r1 = np.random.uniform(0.85, 0.98)
        r2 = np.random.uniform(0.85, 0.98)
        r3 = np.random.uniform(0.85, 0.98)

    future_30 = area * r1
    future_60 = future_30 * r2
    future_90 = future_60 * r3

    return (
        round(future_30, 2),
        round(future_60, 2),
        round(future_90, 2),
        scenario
    )

# ==================================================
# BUILD DATASET
# ==================================================
rows = []

for area in current:
    f30, f60, f90, scenario = simulate_future(area)

    rows.append([
        round(area, 2),
        f30,
        f60,
        f90,
        scenario
    ])

out_df = pd.DataFrame(
    rows,
    columns=[
        "current_area",
        "future_30d",
        "future_60d",
        "future_90d",
        "scenario"
    ]
)

# Shuffle rows
out_df = out_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# ==================================================
# SAVE
# ==================================================
out_df.to_csv(OUTPUT_CSV, index=False)

print("\nSaved:", OUTPUT_CSV)
print(out_df.head())