# Data Decision Science

"""
Evaluierung der Ergebnisse der Signalzeitreihen
"""

# Module
import numpy as np
import pandas as pd
import os
import glob
from scipy import stats

# Variablen
wd_main = os.getcwd()

# Evaluation
# Einlesen der Daten
os.chdir(wd_main + "\\Export")
bootstrap_files = glob.glob("*_bootstraps_*.csv")
bootstrap_files_li = []
for file in bootstrap_files:
    df = pd.read_csv(file, sep=",", header=0, index_col=0)
    df["transition"] = file.split("_")[2] + "_" + file.split("_")[3].split(".")[0]
    df["market"] = file[:3]
    bootstrap_files_li.append(df)
df_bootstrap_files = pd.concat(bootstrap_files_li)

benchmark_files = glob.glob("*_return_rates_*.csv")
benchmark_files_li = []
for file in benchmark_files:
    df = pd.read_csv(file, sep=",", header=0, index_col=0)
    df["transition"] = file.split("_")[4] + "_" + file.split("_")[5].split(".")[0]
    df["market"] = file[:3]
    benchmark_files_li.append(df)
df_benchmark_files = pd.concat(benchmark_files_li)

# Speichern des Zwischenstands
os.chdir(wd_main + "\\Results")
df_bootstrap_files = df_bootstrap_files.reset_index().drop("index", axis=1)
df_bootstrap_files.to_csv("df_bootstrap_files.csv", sep=",", header=True)
df_benchmark_files.to_csv("df_benchmark_files.csv", header=True)

# Benchmark Ergebnisse
stats_bench_dict = {}
for market in df_benchmark_files["market"].unique():
    for transition in df_benchmark_files["transition"].unique():
        df_selected = df_benchmark_files[
            (df_benchmark_files["market"] == market)
            & (df_benchmark_files["transition"] == transition)
        ]
        df_selected = df_selected.iloc[:, :7]
        for column in df_selected.columns:
            if transition == "full_transition":
                t_value = np.nan
                p_value = np.nan
                li_selected = [np.mean(df_selected[column]), t_value, p_value]
                name = market + "_" + transition + "_" + column
                stats_bench_dict[name] = li_selected
            else:
                bench_df = df_benchmark_files[
                    (df_benchmark_files["market"] == market)
                    & (df_benchmark_files["transition"] == "full_transition")
                ]
                t_value, p_value = stats.ttest_ind(
                    bench_df[column], df_selected[column]
                )
                li_selected = [np.mean(df_selected[column]), t_value, p_value]
                name = market + "_" + transition + "_" + column
                stats_bench_dict[name] = li_selected
df_stats_bench = pd.DataFrame(stats_bench_dict)
df_stats_bench.index = ["mean", "t_value", "p_value"]
df_stats_bench = df_stats_bench.transpose()
df_stats_bench.to_excel("df_stats_bench.xlsx", header=True)

# Zusammenführen der Werte je Markt und Umsetzung
stats_dict = {}
for market in df_bootstrap_files["market"].unique():
    for transition in df_bootstrap_files["transition"].unique():
        df_market_transition = df_bootstrap_files[
            (df_bootstrap_files["market"] == market)
            & (df_bootstrap_files["transition"] == transition)
        ]
        statistics = df_market_transition.mean()
        stats_dict[market + "_" + transition] = statistics
df_stats = pd.DataFrame(stats_dict)
df_stats.to_csv("bootstrap_results_all_markets.csv", sep=",", header=True)

# Hypothesentest je Markt
significance_dict = {}
for market in df_bootstrap_files["market"].unique():
    for transition in df_bootstrap_files["transition"].unique():
        if transition == "full_transition":
            significance_values = np.empty((1, 56))
            significance_values[:] = np.nan
            significance_dict[market + "_" + transition] = significance_values[0]
        else:
            df_market_transition = df_bootstrap_files[
                (df_bootstrap_files["market"] == market)
                & (df_bootstrap_files["transition"] == transition)
            ]
            df_market_transition = df_market_transition.iloc[:, :56]
            df_benchmark = df_bootstrap_files[
                (df_bootstrap_files["market"] == market)
                & (df_bootstrap_files["transition"] == "full_transition")
            ]
            df_benchmark = df_benchmark.iloc[:, :56]
            stats_li = []
            for column in df_market_transition.columns:
                t_value, p_value = stats.ttest_ind(
                    df_benchmark[column], df_market_transition[column]
                )
                if t_value > 0:
                    if p_value <= 0.01:
                        value = "*** (+)"
                        stats_li.append(value)
                    elif p_value <= 0.05:
                        value = "** (+)"
                        stats_li.append(value)
                    elif p_value <= 0.1:
                        value = "* (+)"
                        stats_li.append(value)
                    else:
                        stats_li.append(np.nan)
                else:
                    if p_value <= 0.01:
                        value = "*** (-)"
                        stats_li.append(value)
                    elif p_value <= 0.05:
                        value = "** (-)"
                        stats_li.append(value)
                    elif p_value <= 0.1:
                        value = "* (-)"
                        stats_li.append(value)
                    else:
                        stats_li.append(np.nan)
            significance_dict[market + "_" + transition] = stats_li

significance_df = pd.DataFrame(significance_dict)
significance_df.index = df_bootstrap_files.iloc[:, :56].columns
df_stats_with_significance = df_stats.astype(str) + significance_df.fillna("")
df_stats_with_significance.to_csv("df_stats_with_significance.csv", header=True)
df_stats_with_significance_basispunkte = df_stats * 10000
df_stats_with_significance_basispunkte = df_stats_with_significance_basispunkte.round(3)
df_stats_with_significance_basispunkte = df_stats_with_significance_basispunkte.astype(
    str
) + significance_df.fillna("")
df_stats_with_significance_basispunkte.to_csv(
    "df_stats_with_significance_basispunkte.csv", header=True
)


# Hypothesentest über alle Bootstraps hinweg
df_hypothesis_test = df_bootstrap_files[
    df_bootstrap_files["transition"] == "full_transition"
].mean()
df_hypothesis_test = pd.DataFrame(df_hypothesis_test)
df_hypothesis_test.columns = ["full_transition"]
df_hypothesis_test["linear_2d"] = df_bootstrap_files[
    df_bootstrap_files["transition"] == "linear_2d"
].mean()
df_hypothesis_test["linear_3d"] = df_bootstrap_files[
    df_bootstrap_files["transition"] == "linear_3d"
].mean()
df_hypothesis_test["progressive_2d"] = df_bootstrap_files[
    df_bootstrap_files["transition"] == "progressive_2d"
].mean()
df_hypothesis_test["progressive_3d"] = df_bootstrap_files[
    df_bootstrap_files["transition"] == "progressive_3d"
].mean()


def hypothesis_test(column_name, transition):
    """Funktion, die anhand des Spaltennamens einen Hypothesentest durchführt und die dazugehörigen
    t- und p-values als tuple ausgibt."""
    column_name_elements = column_name.split("_")
    if column_name_elements[0] == "buy":
        return (np.nan, np.nan)
    else:
        a = df_bootstrap_files[df_bootstrap_files["transition"] == transition][
            column_name
        ]
        b = df_bootstrap_files[df_bootstrap_files["transition"] == "full_transition"][
            column_name
        ]
        t_value, p_value = stats.ttest_ind(a, b)
        return (t_value, p_value)


for transition in ["linear_2d", "linear_3d", "progressive_2d", "progressive_3d"]:
    column_name = transition + "_test_values"
    df_hypothesis_test[column_name] = [
        hypothesis_test(column, transition) for column in df_hypothesis_test.index
    ]


def significance(test_values):
    """Funktion, die das Signifikanzniveau ausgibt."""
    p_value = test_values[1]
    t_value = test_values[0]
    if t_value > 0:
        if p_value <= 0.01:
            return "*** (+)"
        elif p_value <= 0.05:
            return "** (+)"
        elif p_value <= 0.1:
            return "* (+)"
        else:
            return np.nan
    elif t_value < 0:
        if p_value <= 0.01:
            return "*** (-)"
        elif p_value <= 0.05:
            return "** (-)"
        elif p_value <= 0.1:
            return "* (-)"
        else:
            return np.nan


for transition in ["linear_2d", "linear_3d", "progressive_2d", "progressive_3d"]:
    test_value_column = transition + "_test_values"
    significance_column = transition + "significance"
    df_hypothesis_test[significance_column] = df_hypothesis_test[test_value_column]
    df_hypothesis_test[significance_column] = df_hypothesis_test[
        significance_column
    ].apply(significance)


df_hypothesis_test.to_csv("df_hypothesis_test_bootstrap_all_markets.csv", header=True)
