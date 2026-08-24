import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

df = pd.read_excel("final_data/arsenal_ucl_fotmob_stats (1).xlsx")

df["position"] = df["position"].replace({"Lb": "LB"})

df = df.dropna(subset=["fan_rating"]).copy()

df["position_group"] = np.select(
    [
        df["position"].isin(["CB", "LB", "RB"]),
        df["position"].isin(["CM", "CAM", "CDM"]),
        df["position"].isin(["LW", "RW", "ST"])
    ],
    [
        "Defender",
        "Midfielder",
        "Attacker"
    ],
    default="Unknown"
)

df = df[df["position_group"] != "Unknown"].copy()

per90 = [
    "goals",
    "assists",
    "xg",
    "xa",
    "chances_created",
    "passes",
    "tackles",
    "interceptions",
    "clearances",
    "blocks",
    "recoveries",
    "shots"
]

df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce")

minutes = df["minutes"].replace(0, np.nan)

for column in per90:
    df[column] = pd.to_numeric(df[column], errors="coerce")
    df[column + "_per90"] = df[column] / minutes * 90

candidate_features = {
    "Defender": [
        "goals_per90",
        "assists_per90",
        "passes_per90",
        "tackles_per90",
        "interceptions_per90",
        "clearances_per90",
        "blocks_per90",
        "recoveries_per90",
        "ground_duels_percentage",
        "aerial_duels_percentage"
    ],
    "Midfielder": [
        "goals_per90",
        "assists_per90",
        "xg_per90",
        "xa_per90",
        "chances_created_per90",
        "passes_per90",
        "tackles_per90",
        "interceptions_per90",
        "recoveries_per90",
        "ground_duels_percentage",
        "dribble_percentage"
    ],
    "Attacker": [
        "goals_per90",
        "assists_per90",
        "xg_per90",
        "xa_per90",
        "chances_created_per90",
        "shots_per90",
        "dribble_percentage",
        "ground_duels_percentage"
    ]
}

for column_list in candidate_features.values():
    for column in column_list:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        ).fillna(0)

df["fan_rating"] = pd.to_numeric(
    df["fan_rating"],
    errors="coerce"
)

weights = {}

for group, features in candidate_features.items():

    group_df = df[
        df["position_group"] == group
    ].copy()

    X = group_df[features]
    y = group_df["fan_rating"]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    model = Lasso(
        alpha=0.01,
        positive=True,
        max_iter=100000
    )

    model.fit(
        X_scaled,
        y
    )

    coefficients = model.coef_

    positive_coefficients = np.maximum(
        coefficients,
        0
    )

    total = positive_coefficients.sum()

    if total == 0:
        raise ValueError(
            f"No positive coefficients were learned for {group}"
        )

    normalised_weights = (
        positive_coefficients / total
    )

    weights[group] = {
        feature: weight
        for feature, weight
        in zip(features, normalised_weights)
        if weight > 0
    }

print("=" * 70)
print("TRAINED POSITION-SPECIFIC WEIGHTS")
print("=" * 70)

for group, group_weights in weights.items():

    print()
    print(group.upper())
    print()

    for feature, weight in sorted(
        group_weights.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(
            f"{feature:<35}"
            f"{weight:.4f}"
        )

print()
print("=" * 70)
print("Training Data ")
print("=" * 70)

print("Rows used:", len(df))
print()
print(df["position_group"].value_counts())

print()
print("=" * 70)
print("Weights produced for the final model")
print("=" * 70)

for group, group_weights in weights.items():

    print()
    print(f'"{group}": {{')

    for feature, weight in group_weights.items():

        print(
            f'    "{feature}": '
            f'{weight:.3f},'
        )

    print("}")
