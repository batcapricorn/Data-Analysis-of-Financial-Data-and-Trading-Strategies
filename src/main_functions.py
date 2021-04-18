# Data Decision Science

"""
Implementierung verschiedener Funktionen, um das Timing der Umsetzung 
von technischen Anlagestrategien untersuchen zu können. 
"""

# Module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time
from datetime import timedelta
import statsmodels.formula.api as smf
import math

# Funktionen

# Performance
# Rendite
def rate_of_return(series, kind="normal"):
    """Funktion, die die Renditen einer Series berechnet und
    ebenfalls als Pandas Series zurückgibt. Falls kind='normal' (default) wird die
    normale, falls kind='log' die logarithmierte Rendite ausgegeben."""
    return_rates = series.pct_change()
    if kind == "log":
        return_rates = np.log(1 + return_rates)
    return_rates = return_rates.fillna(0)
    return return_rates


# CAPM-Alpha
def calc_alpha(returns, risk_premium):
    """Funktion, anhand dessen das CAPM-Aplha einer Zeitreihe berechnet werden kann. Hierzu muss zum einen die Zeitreihe der jeweiligen Renditen
    sowie die zugehörigen Risk Premiums angegeben werden."""
    df_alpha = pd.DataFrame(returns)
    df_alpha.columns = ["rate_of_returns"]
    df_alpha["market_risk_premium"] = risk_premium
    Regression = smf.ols(
        formula="rate_of_returns ~ market_risk_premium", data=df_alpha
    )  # Modell
    regoutput = Regression.fit()  # Regression
    alpha = regoutput.params["Intercept"]  # Ausgeben des Alphas
    return alpha


# Sharpe Ratio
def sharpe_ratio(return_rates):
    """Funktion, die die Sharpe Ratio einer Series zurückgibt."""
    sharpe_ratio = np.mean(return_rates) / (np.std(return_rates) ** (0.5))
    return sharpe_ratio


# Bootstraps
# Erstellen der Bootstraps
def bootstrap(series, return_rates, number_replicates=10, chunksize=0, seed=None):
    """Funktion, die die angegebene Anzahl von Bootsraps für die Markt-Series zurückgibt.
    Hierbei wird berücksichtigt, dass es sich um zeitlich abhängige Arrays handelt. Als Input
    muss die Zeitreihe sowie eine Series der dazugehörigen Renditen, anhand dessen der Bootstrap
    erstellt wird, angegeben werden. Des Weiteren kann die Anzahl der Straps (Default=10), die
    Chunksize (Default=0, also keine Chunks) und die Seed-Zahl bestimmt werden."""
    if seed != None:
        random.seed(seed)
    bootstraps = []
    for iteration in range(number_replicates):
        bootstrap = [series.fillna(method="bfill")[0]]
        chunk = return_rates
        for observation in range(len(series) - 1):
            if chunksize != 0:
                chunk = return_rates[
                    max(0, (observation - chunksize)) : min(
                        (observation + chunksize), (len(series) - 1)
                    )
                ]
            random_index = random.randint(0, (len(chunk) - 1))
            random_return_rate = chunk[random_index]
            new_observation = bootstrap[-1] * (1 + random_return_rate)
            bootstrap.append(new_observation)
        bootstraps.append(bootstrap)
    return bootstraps


# Graph der Bootstraps
def plot_bootstraps(series, bootstraps, title, market_label, currency="USD"):
    """Funktion, die die erstellten Bootstraps visualisiert und den originalen
    Marktdaten gegenüber stellt. Hierzu muss die origniale Zeitreihe, der
    Bootstrap-Array, der Titel und das Marktlabel angegeben werden. Weiterhin kann
    die Währung bestimmt werden (Default='USD')"""
    fig = plt.figure()
    plt.plot(series, label=market_label)
    sns.despine(fig, left=True, bottom=True)
    counter = 1
    series_df = pd.DataFrame(series)
    for item in bootstraps:
        series_df["strap"] = item
        if counter == 1:
            plt.plot(series_df["strap"], color="grey", label="Bootstraps", alpha=0.4)
            counter += 1
        else:
            plt.plot(series_df["strap"], color="grey", alpha=0.4)
    plt.legend(loc="upper right", edgecolor="white")
    plt.ylabel(currency, fontweight="bold")
    plt.xlabel("Zeit", fontweight="bold")
    plt.title(title, pad=22, fontweight="bold")
    plt.show()
    plt.close()


# Umsetzungen
# Signalwechsel
def detect_signal_change(signals):
    """Funktion, die ausliest, wann das Signal umschlägt."""
    signal_change = signals.pct_change().fillna(0)

    def convert_signal_change(x):
        if x == 0:
            return x
        else:
            return 1

    signal_change = pd.Series([convert_signal_change(x) for x in signal_change])
    signal_change.index = signals.index
    return signal_change


# Gewichtung gemäß Umsetzung
def transition_weights(signals, transitions):
    """Funktion, die ausliest, zu welchen Anteilen jeweils gemäß der Umsetzung in die Strategie
    investiert werden soll."""
    signal_change = detect_signal_change(signals)
    weights = signal_change.copy()
    changes = list(weights[weights == 1].index)
    weights = weights.replace(0, 1)
    for index in changes:
        if 0 not in transitions["days"]:
            weights[index] = 0
        else:
            weights[index] = transitions["portions"][0]
        for i in range(len(transitions["days"])):
            try:
                day_range = transitions["days"][i + 1] - transitions["days"][i]
            except:
                day_range = 1
            if i == 0:
                day_start = index + timedelta(transitions["days"][i])
                if day_start > list(weights.index)[-1]:
                    break
                while day_start not in weights.index:
                    day_start = day_start + timedelta(1)
            for day in range(day_range):
                if (i == 0) and (day == 0):
                    current_date = day_start
                else:
                    try:
                        current_date = current_date + timedelta(1)
                    except:
                        break
                if current_date > list(weights.index)[-1]:
                    break
                while current_date not in weights.index:
                    current_date = current_date + timedelta(1)
                weights[current_date] = transitions["portions"][i]
    zero_weights = signals[signals == 0].index
    weights[zero_weights] = 0
    return weights


# Renditen gemäß Umsetzung
def returns_weights(
    signals, original_returns, transitions, kind="normal", transaction_costs=0
):
    """Funktion, die anhand der Signale, der Renditen und den Umsetzungen die richtige Renditen-Zeitreihe berechnet. Zudem kann angegeben werden,
    ob der Output normale (default) oder log-Renditen enthalten soll. Des Weiteren können prozentuale Transaktionskosten berücksichtigt werden
    (Default=0). Es werden jeweils beide Zeitreihen, mit und ohne Transaktionskosten, ausgegeben."""
    weights = transition_weights(signals, transitions)  # Gewichtungen
    df_weighted_returns = pd.DataFrame(weights, columns=["weights"])
    df_weighted_returns["original_returns"] = original_returns  # Renditen
    if len(transitions["days"]) > 1:
        df_weighted_returns["investments"] = abs(
            df_weighted_returns["weights"].replace(1, 0).shift()
            - df_weighted_returns["weights"].replace(1, 0)
        )  # Wann wie viel prozentual ivestiert wird
    else:
        df_weighted_returns["investments"] = abs(
            df_weighted_returns["weights"].shift() - df_weighted_returns["weights"]
        )  # Falls alles am Folgetag umgesetzt werden soll
    df_weighted_returns["investments"] = df_weighted_returns["investments"].fillna(0)
    df_weighted_returns["signal"] = signals  # Signale
    df_weighted_returns["available"] = (
        1 - df_weighted_returns["weights"]
    )  # Verfügbares Kapital

    df_weighted_returns["investment_gains"] = 0  # Renditen des investierten Kapitals
    counter = 1
    df_weighted_returns_copy = df_weighted_returns.copy().fillna(0)
    for index, row in df_weighted_returns_copy.iterrows():
        if counter == 1:
            previous_value = 1
            counter += 1
            df_weighted_returns.loc[index, "investment_gains"] = previous_value
            continue
        else:
            if row["investments"] != 0:
                previous_value = (previous_value * row["investments"]) * (
                    1 + row["original_returns"] * row["signal"]
                ) + (
                    previous_value * (1 - row["investments"])
                )  # Wert des investierten Kapitals
            else:
                previous_value = (
                    row["signal"] * row["original_returns"] + 1
                ) * previous_value
            if np.isnan(previous_value) == True:
                previous_value = 0
        df_weighted_returns.loc[index, "investment_gains"] = previous_value
    df_weighted_returns["return_transition"] = rate_of_return(
        df_weighted_returns["investment_gains"], kind="normal"
    )  # Rendite
    df_weighted_returns["value_when_invested"] = df_weighted_returns[
        "investment_gains"
    ].copy()
    df_weighted_returns["value_when_invested"][
        df_weighted_returns["investments"] == 0
    ] = 0
    df_weighted_returns["transaction_costs"] = (
        df_weighted_returns["value_when_invested"] * transaction_costs
    )  # Transaktionskosten
    df_weighted_returns["return_transition_with_costs"] = (
        df_weighted_returns["return_transition"]
        - df_weighted_returns["transaction_costs"]
    )
    return (
        df_weighted_returns["return_transition"].replace(math.inf, 0),
        df_weighted_returns["return_transition_with_costs"].replace(math.inf, 0),
    )


# 38/200 Momentum
#% Momentum
def momentum(obs):
    """Funktion, die anhand einer Observation das Momentum bestimmt. Hierbei stellt die Observation eine Differenz
    von relevanten Crossover-Kennzahlen dar."""
    if obs > 0:
        return 1
    elif obs == 0:
        return 0
    elif obs < 0:
        return -1
    else:
        return np.nan

    # Signalberechnung


def crossover_signal(series, avg_short=38, avg_long=200):
    """Funktion, die das Momentum-Modell einer Series anhand einer Crossover-Strategie zurück gibt. Hierbei
    kann die untere  (Default=38) und die obere Durchschnittsgrenze (Default=200) bei Bedarf variiert werden."""
    momentum_df = pd.DataFrame(series.fillna(method="bfill"))
    try:
        momentum_df["avg_short"] = (
            momentum_df[series.name].rolling(avg_short, win_type=None).mean()
        )
        momentum_df["avg_long"] = (
            momentum_df[series.name].rolling(avg_long, win_type=None).mean()
        )
    except:
        momentum_df["avg_short"] = (
            momentum_df.iloc[:, 0].rolling(avg_short, win_type=None).mean()
        )
        momentum_df["avg_long"] = (
            momentum_df.iloc[:, 0].rolling(avg_long, win_type=None).mean()
        )
    momentum_df["difference"] = momentum_df["avg_short"] - momentum_df["avg_long"]
    momentum_df["signal"] = momentum_df["difference"].apply(momentum)
    return momentum_df["signal"]


# Clenow Counter Plunger
# Average True Range
def average_true_range(high_series, low_series, close_series, true_range=20):
    """Funktion, die anhand der Low-, High- und Closing-Prices die Avereage True Range berechnet und als
    Series zurückgibt."""
    dataframe_atr = pd.DataFrame()
    dataframe_atr["ATR1"] = abs(high_series - low_series)
    dataframe_atr["ATR2"] = abs(high_series - close_series.shift())
    dataframe_atr["ATR3"] = abs(low_series - close_series.shift())
    dataframe_atr["TrueRange"] = (
        dataframe_atr[["ATR1", "ATR2", "ATR3"]].max(axis=1).fillna(method="bfill")
    )
    dataframe_atr["AverageTrueRange"] = (
        dataframe_atr["TrueRange"].rolling(true_range).mean()
    )
    return dataframe_atr["AverageTrueRange"]

    # Auswählen des jeweigen besten Readings


def pick_reading(row):
    """Funktion, die anhand eines Trend-Wertes jeweils den Maximal- oder den Minimalwert einer Observation zurückgibt.
    Diese Funktion geht Hand in Hand mit der Funktion "best_reading"."""
    if row["trend"] == 1:
        return row["high"]
    elif row["trend"] == -1:
        return row["low"]
    else:
        return np.nan

    # Ermitteln der besten Readings


def best_reading(high_series, low_series, trend, window=20):
    """Funktion, die anhand der High- und der Low-Prices sowie der Trend-Zeitreihe die jeweils besten Readings
    ermittelt."""
    dataframe_br = pd.DataFrame()
    dataframe_br["high"] = (
        high_series.fillna(method="bfill").rolling(window, win_type=None).max()
    )
    dataframe_br["low"] = (
        low_series.fillna(method="bfill").rolling(window, win_type=None).min()
    )
    dataframe_br["trend"] = trend
    dataframe_br["best_reading"] = dataframe_br.apply(pick_reading, axis=1)
    return dataframe_br["best_reading"]

    # Trend definieren


def calc_trend(obs):
    """Funktion, die anhand einer Observation den Trend bestimmt. Hierbei stellt die Observation eine Differenz
    von relevanten Plunger-Kennzahlen dar."""
    if obs > 0:
        return 1
    elif obs < 0:
        return -1
    else:
        return 0

    # Ermitteln der relevanten Kennzahlen


def clenow_counter_plunger_criteria(
    high_series,
    low_series,
    close_series,
    trend="ma",
    mean_lower=38,
    mean_higher=200,
    window_reading=20,
    true_range=20,
    plunge_trigger=3,
):
    """Funktion, die anhand der Low-, High- und Closing-Prices einen Dataframe zurückgibt, der den Trend, die ATR,
    die besten Readings und die Plunger enthält. Es kann bei Bedarf die Trendstrategie (Default = 'ma')
    sowie die dazugehörigen Unter- und Obergrenzen festgelegt werden (Default 38 bzw. 200). Des Weiteren kann das Fenster der Readings (Default=20) sowie die
    True Range (Default=20) variiert werden. Ebenfalls kann der Plunge Trigger geändert werden, wobei dieser im Normallfall
    sinngemäß 3 beträgt."""
    dataframe_ccp = pd.DataFrame(close_series, columns=["px_last"])  # Closing-Price
    if trend == "ewm":
        dataframe_ccp["trend"] = (
            dataframe_ccp["px_last"].ewm(span=mean_lower, min_periods=mean_lower).mean()
            - dataframe_ccp["px_last"]
            .ewm(span=mean_higher, min_periods=mean_higher)
            .mean()
        )
        dataframe_ccp["trend"] = dataframe_ccp["trend"].apply(calc_trend)  # Trend EWM
    elif trend == "ma":
        dataframe_ccp["trend"] = (
            dataframe_ccp["px_last"].fillna(method="bfill").rolling(mean_lower).mean()
            - dataframe_ccp["px_last"]
            .fillna(method="bfill")
            .rolling(mean_higher)
            .mean()
        )
        dataframe_ccp["trend"] = dataframe_ccp["trend"].apply(momentum)
        dataframe_ccp["trend"] = dataframe_ccp["trend"].fillna(0)
    dataframe_ccp["atr"] = average_true_range(
        high_series, low_series, close_series, true_range=true_range
    )  # ATR
    dataframe_ccp["reading_max"] = (
        high_series.fillna(method="bfill").rolling(window=window_reading).max()
    )
    dataframe_ccp["reading_min"] = (
        low_series.fillna(method="bfill").rolling(window=window_reading).min()
    )
    dataframe_ccp["plunge_up"] = (
        dataframe_ccp["reading_max"] - close_series
    ) / dataframe_ccp["atr"]
    dataframe_ccp["plunge_down"] = (
        close_series - dataframe_ccp["reading_min"]
    ) / dataframe_ccp["atr"]
    dataframe_ccp["plunge_up_flag"] = dataframe_ccp["plunge_up"] >= plunge_trigger
    dataframe_ccp["plunge_down_flag"] = dataframe_ccp["plunge_down"] <= plunge_trigger
    return dataframe_ccp

    # Umwandlung der Kennzahlen


def plunger_signal(df_plunger, confirmation_period=2):
    """Funktion, anhand dessen die Clenow Counter Plunger Kriterien in ein Trading-Signal umgewandelt werden können. Hierbei muss ein
    dementsprechender Dataframe angegeben werden, der die Spalten "trend" (1, -1, 0), "plunge_up_flag"/"plunge_down_flag" (bool), "px_last"
    (Closing-Price, float) sowie "atr" (float). Zudem kann die Confirmation Period variiert werden, welche im Default
    2 beträgt. Diese bezieht sich allerdings nur auf den Exit."""
    df_plunger_signal = df_plunger.copy()
    df_plunger_signal["px_last"] = df_plunger_signal["px_last"].fillna(method="bfill")
    state = 0
    confirmation_days = 0
    stop = np.nan
    target = np.nan
    df_plunger_signal["signal"] = 0
    for index, row in df_plunger_signal.iterrows():
        # Keine Aktivität
        if state == 0:
            # Trigger Long
            if (row["trend"] == 1) & (row["plunge_up_flag"] == True):
                stop = row["px_last"] - 2 * row["atr"]
                target = row["px_last"] + 4 * row["atr"]
                state = 1
                df_plunger_signal.loc[index, "signal"] = 1
            # Trigger Short
            elif (row["trend"] == -1) & (row["plunge_down_flag"] == True):
                stop = row["px_last"] + 2 * row["atr"]
                target = row["px_last"] - 4 * row["atr"]
                state = -1
                df_plunger_signal.loc[index, "signal"] = -1
            else:
                continue
        # Bereits Long
        elif state == 1:
            # Kein Exit
            if (
                (row["trend"] == 1)
                & (row["px_last"] < target)
                & (row["px_last"] > stop)
            ):
                df_plunger_signal.loc[index, "signal"] = 1
            elif (row["trend"] == 1) & (row["plunge_up_flag"] == True):
                df_plunger_signal.loc[index, "signal"] = 1
            # Ansonsten Exit
            else:
                if confirmation_days > confirmation_period:
                    if (row["trend"] == -1) & (row["plunge_down_flag"] == True):
                        stop = row["px_last"] + 2 * row["atr"]
                        target = row["px_last"] - 4 * row["atr"]
                        state = -1
                        confirmation_days = 0
                    else:
                        state = 0
                        confirmation_days = 0
                else:
                    df_plunger_signal.loc[index, "signal"] = 1
                    confirmation_days += 1
        # Bereits Short
        elif state == -1:
            # Kein Exit
            if (
                (row["trend"] == -1)
                & (row["px_last"] > target)
                & (row["px_last"] < stop)
            ):
                df_plunger_signal.loc[index, "signal"] = -1
            elif (row["trend"] == -1) & (row["plunge_down_flag"] == True):
                df_plunger_signal.loc[index, "signal"] = -1
            # Ansonsten Exit
            else:
                if confirmation_days > confirmation_period:
                    if (row["trend"] == 1) & (row["plunge_up_flag"] == True):
                        stop = row["px_last"] - 2 * row["atr"]
                        target = row["px_last"] + 4 * row["atr"]
                        state = 1
                        df_plunger_signal.loc[index, "signal"] = 1
                        confirmation_days = 0
                    else:
                        state = 0
                        df_plunger_signal.loc[index, "signal"] = 0
                        confirmation_days = 0
                else:
                    df_plunger_signal.loc[index, "signal"] = -1
                    confirmation_days += 1
    return df_plunger_signal["signal"]


# Mean Reversion Z-Scores
def rolling_z_score(series, window=180):
    """Funktion, die den rolling z-score einer Series ausgibt. Hierbei ist window im Default 180."""
    col_mean = series.fillna(method="bfill").rolling(window=window).mean()
    col_std = series.fillna(method="bfill").rolling(window=window).std()
    z_scores = (series - col_mean) / col_std
    return z_scores


def z_score_signals(series, window=180, period=50):
    """Funktion, die anhand einer Renditenzeitreihe, dem dazugehörigen Zeitfenster der z-scores(default=180) sowie
    des Investitionszeitraums (default=50) Signale einer Mean Reversion ausgibt."""
    df_z_score = pd.DataFrame(series, columns=["px_last"])
    df_z_score["z_score"] = rolling_z_score(series, window=window)
    day_counter = 0
    state = 0
    df_z_score["signal"] = 0
    for index, row in df_z_score.iterrows():
        # Keine Aktivität
        if state == 0:
            # Trigger Long
            if row["z_score"] < -2:
                state = 1
                df_z_score.loc[index, "signal"] = state
                day_counter += 1
            # Trigger Short
            elif row["z_score"] > 2:
                state = -1
                df_z_score.loc[index, "signal"] = state
                day_counter += 1
            else:
                continue
        # Bereits Long
        elif state == 1:
            # Kein Exit aufgrund des Zeitraums
            if day_counter <= 50:
                df_z_score.loc[index, "signal"] = state
                day_counter += 1
            # Kein Exit aufgrund der z-score
            elif row["z_score"] < -2:
                df_z_score.loc[index, "signal"] = state
                day_counter = 1
            # Wechsel aufgrund der z-score
            elif row["z_score"] > 2:
                state = -1
                df_z_score.loc[index, "signal"] = state
                day_counter = 1
            # Ansonsten Exit
            else:
                state = 0
                df_z_score.loc[index, "signal"] = state
                day_counter = 1
        # Bereits Short
        elif state == -1:
            # Kein Exit aufgrund des Zeitraums
            if day_counter <= 50:
                df_z_score.loc[index, "signal"] = state
                day_counter += 1
            # Kein Exit aufgrund der z-score
            elif row["z_score"] > 2:
                df_z_score.loc[index, "signal"] = state
                day_counter = 1
            # Wechsel aufgrund der z-score
            elif row["z_score"] < -2:
                state = -1
                df_z_score.loc[index, "signal"] = state
                day_counter = 1
            # Ansonsten Exit
            else:
                state = 0
                df_z_score.loc[index, "signal"] = state
                day_counter = 1
    return df_z_score["signal"]
