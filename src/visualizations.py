# Data Decision Science

"""
Skript, in welchem verschiedene Graphen für die PowerPoint-Präsentation
erstellt werden.
"""

# Eigene Module
import main_functions as mf
import exploratory_data_analysis as eda

# Allgemeine Module
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# Directories
wd_main = os.getcwd()
wd_data = wd_main + "\\Data"

# Graphen
#% Bootstraps
#% Einlesen der Marktdaten
os.chdir(wd_data)
df_us1 = pd.read_csv(
    "us1_index.csv", sep=",", header=0, parse_dates=["date"], dayfirst=True
).set_index("date")
df_es1 = pd.read_csv(
    "es1_index.csv", sep=",", header=0, parse_dates=["date"], dayfirst=True
).set_index("date")

#% Erstellen der Bootstraps
bootstraps_us1 = mf.bootstrap(
    series=df_us1["px_last"],
    return_rates=mf.rate_of_return(df_us1["px_last"]),
    number_replicates=10,
    chunksize=180,
    seed=123,
)
bootstraps_es1 = mf.bootstrap(
    series=df_es1["px_last"],
    return_rates=mf.rate_of_return(df_es1["px_last"]),
    number_replicates=10,
    chunksize=180,
    seed=123,
)

#% Plotten der Bootstraps
mf.plot_bootstraps(
    series=df_us1["px_last"],
    bootstraps=bootstraps_us1,
    title="30 Year U.S. T-Bond Future",
    market_label="US1",
)
mf.plot_bootstraps(
    series=df_es1["px_last"],
    bootstraps=bootstraps_es1,
    title="E-Mini S&P 500 Future",
    market_label="ES1",
)


#% Renditenboxplots der Closing-Prices
#% Einlesen der Renditen
value_column = []
market_column = []
for file in glob.glob("*.csv"):
    df = pd.read_csv(
        file, sep=",", header=0, parse_dates=["date"], dayfirst=True
    ).set_index("date")
    return_rates = mf.rate_of_return(df["px_last"])
    value_column.extend(return_rates)
    key_name = file[:3]
    market_list = [key_name for i in range(len(return_rates))]
    market_column.extend(market_list)

    #% Zusammenführen der Daten
df_boxplots = pd.DataFrame()
df_boxplots["return_rates"] = value_column
df_boxplots["market"] = market_column

#% Plot
# fig = plt.figure(figsize=(50,30))
fig = plt.figure(figsize=(20, 15))
ax = plt.axes(frameon=False)
_ = sns.boxplot(y="return_rates", x="market", data=df_boxplots, palette="GnBu")
_ = plt.ylabel("Tagesrendite", fontweight="bold", size="xx-large")
_ = plt.xlabel("Markt", fontweight="bold", size="xx-large")
_ = plt.xticks(fontweight="bold")
_ = plt.title(
    "Boxplots der Tagesrenditen aller Märkte",
    pad=20,
    fontweight="bold",
    size="xx-large",
)
ax.get_xaxis().tick_bottom()
ax.axes.get_yaxis().set_visible(True)
plt.show()
plt.close()

#% Kursverläufe anhand der Closing-Prices
#% Auslesen der Daten
value_column = []
market_column = []
date_column = []
for file in glob.glob("*.csv"):
    df = pd.read_csv(
        file, sep=",", header=0, parse_dates=["date"], dayfirst=True
    ).set_index("date")
    value_column.extend(df["px_last"])
    date_column.extend(df.index)
    key_name = file[:3]
    market_list = [key_name for i in range(len(df["px_last"]))]
    market_column.extend(market_list)

    #% Zusammenführen der Daten
df_closing_prices = pd.DataFrame()
df_closing_prices["px_last"] = value_column
df_closing_prices["market"] = market_column
df_closing_prices["date"] = date_column

#% Plot
fig = plt.figure(figsize=(50, 30))
ax = plt.axes(frameon=False)
_ = sns.lineplot(
    x="date", y="px_last", hue="market", data=df_closing_prices, palette="GnBu"
)
_ = plt.ylabel("Kurswert in USD", fontweight="bold", size="xx-large")
_ = plt.xlabel("Zeit", fontweight="bold", size="xx-large")
_ = plt.xticks(fontweight="bold")
_ = plt.title("Kursverläufe aller Märkte", pad=20, fontweight="bold", size="xx-large")
_ = plt.legend(loc="upper right", fontsize="xx-large")
ax.get_xaxis().tick_bottom()
ax.axes.get_yaxis().set_visible(True)
plt.show()
plt.close()

#% Signale
#% Crossover
#% Zusammenführen der Daten
df_crossover_us1 = pd.DataFrame()
df_crossover_us1["Signal"] = pd.Categorical(
    df_us1["crossover_signal"]
    .replace(1, "Long")
    .replace(0, "Kein Signal")
    .replace(-1, "Short")
)
df_crossover_us1["Kurswert"] = df_us1["px_last"]
df_crossover_us1.index = df_us1.index
df_crossover_us1 = df_crossover_us1.dropna()

#% Plot
fig = plt.figure(figsize=(20, 10))
ax = plt.axes(frameon=False)
_ = sns.scatterplot(
    x=df_crossover_us1.index,
    y="Kurswert",
    hue="Signal",
    palette="viridis",
    data=df_crossover_us1,
    marker="x",
)
_ = plt.legend(loc="upper right", edgecolor="white")
_ = plt.ylabel("Kurswert", fontweight="bold")
_ = plt.xlabel("Zeit", fontweight="bold")
_ = plt.title(
    "30 Year U.S. T-Bond Future - Crossover Strategie", pad=20, fontweight="bold"
)
plt.show()
plt.close()

#% Countertrend
#% Zusammenführen der Daten
df_crossover_us1["Signal"] = pd.Categorical(
    df_us1["countertrend_signal"]
    .replace(1, "Long")
    .replace(0, "Kein Signal")
    .replace(-1, "Short")
)

#% Plot
fig = plt.figure(figsize=(20, 10))
ax = plt.axes(frameon=False)
_ = sns.scatterplot(
    x=df_crossover_us1.index,
    y="Kurswert",
    hue="Signal",
    palette="viridis",
    data=df_crossover_us1,
    marker="x",
)
_ = plt.legend(loc="upper right", edgecolor="white")
_ = plt.ylabel("Kurswert", fontweight="bold")
_ = plt.xlabel("Zeit", fontweight="bold")
_ = plt.title(
    "30 Year U.S. T-Bond Future - Countertrend Strategie", pad=20, fontweight="bold"
)
plt.show()
plt.close()
