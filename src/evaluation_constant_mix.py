# Data Decision Science

"""
Evaluierung der Ergebnisse der Constant Mix Strategie
"""

# Module
import numpy as np
import pandas as pd
import os
import glob
from scipy import stats
from scipy.stats.mstats import gmean

# Eigene Module
import main_functions as mf

# Variablen
wd_main = os.getcwd()

# Evaluation
# Einlesen der Export-Datei
os.chdir(wd_main + "\\Daten")
df_results = pd.read_excel("Renditezeitreihen CM.xlsx", header=0)
df_results["date"] = df_results["date"].apply(
    lambda x: pd.to_datetime(x, dayfirst=True)
)
df_results = df_results.set_index("date")

# Data Processing
df_results = df_results.replace(0, 1)
df_results = df_results - 1

# Einlesen der Risk Premiums
os.chdir(wd_main + "\\Other_Data")
df_risk_premium = pd.read_csv(
    "Risk_Premium.csv", header=0, index_col="date", parse_dates=True, dayfirst=True
)
df_risk_premium = df_risk_premium.loc[df_results.index, :]
df_risk_premium = df_risk_premium.fillna(method="bfill")
risk_premium = df_risk_premium["Mkt-RF"]

# Performancemaße
results_dict = {}
for column in df_results.columns:
    # Allgemeine Performancemaße
    sr = mf.sharpe_ratio(df_results[column])
    alpha = mf.calc_alpha(df_results[column], risk_premium)
    std = np.std(df_results[column])
    mean = np.mean(df_results[column])
    g_mean = gmean((df_results[column] + 1).fillna(1)) - 1
    median = np.percentile(df_results[column], 50)
    p25 = np.percentile(df_results[column], 25)
    p75 = np.percentile(df_results[column], 75)

    # Signifikanzniveaus
    column_elements = column.split("_")
    if column_elements[0] == "full":
        mean_t_value = np.nan
        mean_p_value = np.nan
    else:
        if column_elements[-1] == "return":
            mean_t_value, mean_p_value = stats.ttest_ind(
                df_results["full_total_return"], df_results[column]
            )
        else:
            mean_t_value, mean_p_value = stats.ttest_ind(
                df_results["full_total_return-tk"], df_results[column]
            )

        # Zusammenführen der Ergebnisse
    results_li = [
        sr,
        alpha,
        std,
        mean,
        g_mean,
        median,
        p25,
        p75,
        mean_t_value,
        mean_p_value,
    ]
    results_dict[column] = results_li

df_results_hypothesistest = pd.DataFrame(results_dict)

df_results_hypothesistest.index = [
    "sr",
    "alpha",
    "std",
    "mean",
    "g_mean",
    "median",
    "p25",
    "p75",
    "mean_t_value",
    "mean_p_value",
]


# Speichern der Ergebnisse
df_results_hypothesistest.to_excel("constant_mix_significance.xlsx", header=True)
