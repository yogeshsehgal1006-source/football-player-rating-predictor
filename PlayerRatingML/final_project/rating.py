import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# 1. Import my excel spreadsheet into Python and turn it into a dataframe
df = pd.read_excel("final_data/arsenal_ucl_fotmob_stats (1).xlsx")

position_map = {  # 2. Creating a variable to group Defender Positions into Defenders, Midfielder Positions into Midfielders and Attacker Positions into Attackers
    "CB": "Defender",
    "LB": "Defender",
    "RB": "Defender",
    "CM": "Midfielder",
    "CDM": "Midfielder",
    "CAM": "Midfielder",
    "LW": "Attacker",
    "RW": "Attacker",
    "ST": "Attacker"
}


# In the position column, remove any spaces and convert everything into Uppercase
df["position"] = df["position"].str.strip().str.upper()
# Look at every value in position column and map it into the corresponding position map
df["position_group"] = df["position"].map(position_map)
# Drop the null values in the fan rating column as some data is missing
df = df.dropna(subset=["fan_rating", "fotmob_rating", "position_group"]).copy()

features = [  # Choose a set of statistics which i think will create the best prediction for our machine learning model
    "minutes", "goals", "assists", "xg", "xa", "chances_created", "passes",
    "tackles", "interceptions", "clearances", "blocks", "recoveries",
    "ground_duels_percentage", "aerial_duels_percentage", "shots",
    "dribble_percentage"
]


for c in features + ["fotmob_rating", "fan_rating"]:
    # Turn everything into numerical values
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Drop any rows where rating is missing
df = df.dropna(subset=["fotmob_rating", "fan_rating", "position_group"]).copy()

per90 = [  # Use per 90 statistics rather then individual game statistics as this creates fairer date
    "goals", "assists", "xg", "xa", "chances_created", "passes", "tackles",
    "interceptions", "clearances", "blocks", "recoveries", "shots"
]


minutes = df["minutes"].replace(0, np.nan)  # Replace 0 with nan

for c in per90:
    df[c+"_per90"] = df[c] / minutes * 90

per90_features = [  # Average out the selected features by per 90
    "goals_per90", "assists_per90", "xg_per90", "xa_per90",
    "chances_created_per90", "passes_per90", "tackles_per90",
    "interceptions_per90", "clearances_per90", "blocks_per90",
    "recoveries_per90", "shots_per90", "ground_duels_percentage",
    "aerial_duels_percentage", "dribble_percentage"
]

df[per90_features] = df[per90_features].replace(  # Replace any infinite values with nan and then replaces this with 0
    [np.inf, -np.inf], np.nan).fillna(0)


weights = {  # Weights from the training data
    "Defender": {
        "goals_per90": 0.3894,
        "clearances_per90": 0.3505,
        "ground_duels_percentage": 0.1259,
        "tackles_per90": 0.0760,
        "assists_per90": 0.0483,
        "blocks_per90": 0.0100
    },

    "Midfielder": {
        "interceptions_per90": 0.2347,
        "xg_per90": 0.2333,
        "goals_per90": 0.2071,
        "assists_per90": 0.1684,
        "tackles_per90": 0.1042,
        "chances_created_per90": 0.0479,
        "passes_per90": 0.0043
    },

    "Attacker": {
        "goals_per90": 0.5307,
        "shots_per90": 0.2406,
        "assists_per90": 0.1201,
        "dribble_percentage": 0.0728,
        "ground_duels_percentage": 0.0358
    }
}


df["statistical_score"] = 0.0  # Initially everyone starts with a rating of 0

for group, group_weights in weights.items():  # Loop through each position group
    # Boolean filter to isolate rows only belonging to the current position group
    mask = df["position_group"] == group
    cols = list(group_weights)
    # Centralizes the data, so that stats are measured on the same scale
    scaler = StandardScaler()
    # Convert into z scores i.e +1 is one standard deviation above the average and -1 is one standard deviation below the average
    z = scaler.fit_transform(df.loc[mask, cols])
    score = np.zeros(z.shape[0])  # Start off with an empty score of 0
    for i, c in enumerate(cols):  # Loop through the statistics one at a time
        # Take the standard statisitc and multiply it by the weight assigned to that position and category
        score += z[:, i] * group_weights[c]
    # Put the new score back into the dataframe
    df.loc[mask, "statistical_score"] = score

df["statistical_rating"] = (
    7.2 + .65 *
    # Sees how far or below the score is to the average score
    (df["statistical_score"] - df["statistical_score"].mean())
    / df["statistical_score"].std()
)

# Create a lambda to see how much control the statisctical rating should have compared to fotmob
lambdas = np.arange(0, 1.01, .05)
# Divide data into 5 sets to provide a more accurate way of how my mdeol will do on unseen data
kf = KFold(n_splits=5, shuffle=True, random_state=42)

lambda_rows = []

for lam in lambdas:  # Test every lambda value i.e 0.05, 0.10, 0.15
    scores = (1-lam)*df["fotmob_rating"] + lam * \
        df["statistical_rating"]  # Creating the combined rating
    maes, r2s = [], []

    for train, test in kf.split(df):  # Loop through the 5 folds
        correction = (  # Calculating the difference between the average fan rating and the average combined model rating
            df["fan_rating"].iloc[train].mean()
            - scores.iloc[train].mean()
        )
        # Taking the combined score for the unseen data and add the correction
        pred = scores.iloc[test] + correction
        # Take the fan rating and predicted rating and find the difference, we want a low MAE
        maes.append(mean_absolute_error(df["fan_rating"].iloc[test], pred))
        # Measures how well the predictions explain variations in fan ratings, we want a high value.
        r2s.append(r2_score(df["fan_rating"].iloc[test], pred))

    lambda_rows.append({  # Calculates the average after the 5 fold has been done
        "lambda": lam,
        "CV_MAE": np.mean(maes),  # MAE Average
        "CV_MAE_STD": np.std(maes),  # How much the MAE varies between folds
        "CV_R2": np.mean(r2s)  # R squared average
    })

# Turns the results into a dataframe
lambda_results = pd.DataFrame(lambda_rows)
# Find the lambda with the lowest MAE
best_lambda = lambda_results.loc[lambda_results["CV_MAE"].idxmin(), "lambda"]

df["final_rating"] = (  # Can create the final rating as we now have the best lambda
    (1-best_lambda)*df["fotmob_rating"]
    + best_lambda*df["statistical_rating"]
)

# Works out the difference between the fotmob rating and the predicted rating
df["fotmob_adjustment"] = df["final_rating"] - df["fotmob_rating"]

# Measures accuracy of fotmob ratings with fan ratings
fotmob_mae = mean_absolute_error(df["fan_rating"], df["fotmob_rating"])
fotmob_r2 = r2_score(df["fan_rating"], df["fotmob_rating"])
final_mae = mean_absolute_error(
    df["fan_rating"], df["final_rating"])  # Measures the final model
final_r2 = r2_score(df["fan_rating"], df["final_rating"])

# Based on the best Lambda, then gets its corresponding MAE etc
best = lambda_results[lambda_results["lambda"] == best_lambda].iloc[0]

print("="*70)
print("Final Fotmob + Statistical fan rating model")
print("="*70)
print("Rows used:", len(df))
print(df["position_group"].value_counts())
print()
print("Best lambda:", best_lambda)
print("FotMob weight:", 1-best_lambda)
print("Statistical weight:", best_lambda)
print()
print("FotMob MAE:", fotmob_mae)
print("FotMob R²:", fotmob_r2)
print("Final MAE:", final_mae)
print("Final R²:", final_r2)
print("CV MAE:", best["CV_MAE"])
print("CV MAE standard deviation:", best["CV_MAE_STD"])
print("CV R²:", best["CV_R2"])
print("MAE improvement:", fotmob_mae-final_mae)
print("Percentage improvement:", (fotmob_mae-final_mae)/fotmob_mae*100)
print()
print("Final Weights for each position")
for group, group_weights in weights.items():
    print("\n"+group.upper())
    for stat, weight in group_weights.items():
        print(f"{stat:<35}{weight:.3f}")

results = df[  # Creates our results table
    ["match_id", "opponent", "player", "position", "position_group", "fotmob_rating",
     "fan_rating", "statistical_rating", "final_rating",
     "fotmob_adjustment"]
    # Sorts players from highest to lowest, based on how much the adjustments differed.
].sort_values("fotmob_adjustment", ascending=False)

positive = results.head(15)  # Top 15 adjustments
negative = results.sort_values("fotmob_adjustment").head(
    15)  # Top 15 negative adjustments

with pd.ExcelWriter("Fan_Rating_Model.xlsx", engine="openpyxl") as writer:
    # Creates a final rating sheet in excel
    results.to_excel(writer, sheet_name="Final Ratings", index=False)
    # Creates a positive adjustment sheet
    positive.to_excel(writer, sheet_name="Positive Adjustments", index=False)
    # Creates a negative adjustment sheet
    negative.to_excel(writer, sheet_name="Negative Adjustments", index=False)
    lambda_results.to_excel(
        # Lambda comparison sheet to compare the lamdas and see how each lambda effects the ratings
        writer, sheet_name="Lambda Comparison", index=False)

    summary = pd.DataFrame({  # Creating a small summary of key results
        "Metric": [
            "Best Lambda", "FotMob Weight", "Statistical Weight",
            "FotMob MAE", "Final MAE", "CV MAE", "CV MAE STD",
            "FotMob R²", "Final R²", "CV R²", "MAE Improvement",
            "Percentage Improvement"
        ],
        "Value": [
            best_lambda, 1-best_lambda, best_lambda,
            fotmob_mae, final_mae, best["CV_MAE"], best["CV_MAE_STD"],
            fotmob_r2, final_r2, best["CV_R2"],
            fotmob_mae-final_mae,
            (fotmob_mae-final_mae)/fotmob_mae*100
        ]
    })
    summary.to_excel(writer, sheet_name="Model Summary", index=False)

    weight_rows = [  # Saving specific position weights
        {"position_group": g, "statistic": s, "weight": w}
        for g, ws in weights.items()
        for s, w in ws.items()
    ]
    pd.DataFrame(weight_rows).to_excel(
        writer, sheet_name="Position Weights", index=False
    )

lambda_results.to_excel("Lambda_Comparison.xlsx", index=False)
