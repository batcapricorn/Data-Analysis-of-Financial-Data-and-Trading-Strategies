# Data Decision Science

"""
Timimng-Analyse hinsichtlich der Umsetzung von einer Constant Mix
Strategie. Hierbei wird im 30-Tage Rhythmus umgeschichtet.
Beim Ausführen muss das dementsprechende Working Directory angepasst 
werden
Die Ursprungsdatei wurde manuell zusammen gefügt und enthält die Zeit-
reihen des US1 und ES1-Index.
"""

# Module
import numpy as np
import pandas as pd

# Einlesen der Daten
df_main = pd.read_csv(
    r"C:\Users\Mayer\Desktop\Data Science Ohkrin\Datencm.csv", sep=";"
)

# Linear 1 day
df = df_main.copy()

# Signal
df["Signal"] = 0
df.iloc[::30]["Signal"] = 1

# Hilfsvariablen
df["EQ USD"] = 0
df["T USD"] = 0
df["USD ALL"] = 0
df["USD Traded"] = 0
df["% Traded"] = 0
df["total_return"] = 0
df["total_return-tk"] = 0

# Set up
df.loc[df["Signal"] == 1, "EQ USD"] = 0.6
df.loc[df["Signal"] == 1, "T USD"] = 0.4
df = df.fillna(method="bfill")
df = df.drop(
    ["EQ OPEN", "EQ HIGH", "EQ LOW", "T OPEN", "T HIGH", "T LOW", "Währung"], axis=1
)
df["eq_return"] = df["EQ CLOSE"].pct_change()
df["t_return"] = df["T CLOSE"].pct_change()
df["eq_return_+1"] = df["eq_return"] + 1
df["t_return_+1"] = df["t_return"] + 1
df["eq_return_+1"].fillna(1, inplace=True)
df["t_return_+1"].fillna(1, inplace=True)

# Berechnung der Hilfsvariablen
for i in range(1, len(df)):
    if df.loc[i, "Signal"] == 0:
        df.loc[i, "EQ USD"] = df.loc[i - 1, "EQ USD"] * df.loc[i, "eq_return_+1"]

        df.loc[i, "T USD"] = df.loc[i - 1, "T USD"] * df.loc[i, "t_return_+1"]

        df["USD ALL"] = df["EQ USD"] + df["T USD"]
        df["ratio eq"] = df["EQ USD"] / (df["USD ALL"])
        df["ratio t"] = df["T USD"] / df["USD ALL"]

        if df.loc[i, "ratio eq"] > 0.6:
            df.loc[i, "USD Traded"] = df.loc[i, "EQ USD"] - 0.6 * df.loc[i, "USD ALL"]

        if df.loc[i, "ratio t"] > 0.4:
            df.loc[i, "USD Traded"] = df.loc[i, "T USD"] - 0.4 * df.loc[i, "USD ALL"]

        df.loc[i, "% Traded"] = df.loc[i, "USD Traded"] / df.loc[i, "USD ALL"]

    if df.loc[i, "Signal"] == 1:

        df.loc[i, "USD Traded"] = df.loc[i - 1, "USD Traded"]
        df.loc[i, "% Traded"] = df.loc[i - 1, "% Traded"]

# Combined return
for i in range(1, len(df)):
    df.loc[i, "total_return"] = (
        df.loc[i - 1, "ratio eq"] * df.loc[i, "eq_return_+1"]
        + df.loc[i - 1, "ratio t"] * df.loc[i, "t_return_+1"]
    )

# Combined return - transk
for i in range(1, len(df)):
    if df.loc[i, "Signal"] == 0:
        df.loc[i, "total_return-tk"] = (
            df.loc[i - 1, "ratio eq"] * df.loc[i, "eq_return_+1"]
            + df.loc[i - 1, "ratio t"] * df.loc[i, "t_return_+1"]
        )
    if df.loc[i, "Signal"] != 0:
        df.loc[i, "total_return-tk"] = (
            df.loc[i - 1, "ratio eq"] * df.loc[i, "eq_return_+1"]
            + df.loc[i - 1, "ratio t"] * df.loc[i, "t_return_+1"]
            - df.loc[i, "% Traded"] * 0.015
        )

# Speichern des Zwischenstandes
df.to_excel(
    r"C:\Users\Mayer\Desktop\Data Science Ohkrin\python\Daten\1 Tag standard.xlsx"
)

# Linear 2 day
df = df_main.copy()

# Signale
df["Signal"] = 0
df.iloc[::30]["Signal"] = 1
for i in range(1, len(df)):
    if df.loc[i, "Signal"] == 1:
        df.loc[i + 1, "Signal"] = 2

# Hilfsvariablen
df["EQ USD"] = 0
df["T USD"] = 0
df["USD ALL"] = 0
df["USD Traded"] = 0
df["% Traded"] = 0
df["total_return"] = 0
df["total_return-tk"] = 0
df.loc[0, "EQ USD"] = 0.6
df.loc[0, "T USD"] = 0.4
df = df.fillna(method="bfill")
df = df.drop(
    ["EQ OPEN", "EQ HIGH", "EQ LOW", "T OPEN", "T HIGH", "T LOW", "Währung"], axis=1
)
df["eq_return"] = df["EQ CLOSE"].pct_change()
df["t_return"] = df["T CLOSE"].pct_change()
df["eq_return_+1"] = df["eq_return"] + 1
df["t_return_+1"] = df["t_return"] + 1
df["eq_return_+1"].fillna(1, inplace=True)
df["t_return_+1"].fillna(1, inplace=True)

# Investierte USD
for i in range(1, len(df)):
    if df.loc[i, "Signal"] == 0:
        df.loc[i, "EQ USD"] = df.loc[i - 1, "EQ USD"] * df.loc[i, "eq_return_+1"]

        df.loc[i, "T USD"] = df.loc[i - 1, "T USD"] * df.loc[i, "t_return_+1"]

        df["USD ALL"] = df["EQ USD"] + df["T USD"]
        df["ratio eq"] = df["EQ USD"] / (df["USD ALL"])
        df["ratio t"] = df["T USD"] / df["USD ALL"]

        if df.loc[i, "ratio eq"] > 0.6:
            df.loc[i, "USD Traded"] = df.loc[i, "EQ USD"] - 0.6 * df.loc[i, "USD ALL"]

        if df.loc[i, "ratio t"] > 0.4:
            df.loc[i, "USD Traded"] = df.loc[i, "T USD"] - 0.4 * df.loc[i, "USD ALL"]

        df.loc[i, "% Traded"] = df.loc[i, "USD Traded"] / df.loc[i, "USD ALL"]

    if df.loc[i, "Signal"] == 1:
        df.loc[i, "USD Traded"] = df.loc[i - 1, "USD Traded"]
        df.loc[i, "% Traded"] = df.loc[i - 1, "% Traded"]
        if df.loc[i - 1, "ratio eq"] > 0.6:
            df.loc[i, "EQ USD"] = (
                df.loc[i - 1, "EQ USD"] * df.loc[i, "eq_return_+1"]
                - 1 / 2 * df.loc[i - 1, "USD Traded"]
            )
            df.loc[i, "T USD"] = (
                df.loc[i - 1, "T USD"] * df.loc[i, "t_return_+1"]
                + 1 / 2 * df.loc[i - 1, "USD Traded"]
            )
        if df.loc[i - 1, "ratio t"] > 0.4:
            df.loc[i, "EQ USD"] = (
                df.loc[i - 1, "EQ USD"] * df.loc[i, "eq_return_+1"]
                + 1 / 2 * df.loc[i - 1, "USD Traded"]
            )
            df.loc[i, "T USD"] = (
                df.loc[i - 1, "T USD"] * df.loc[i, "t_return_+1"]
                - 1 / 2 * df.loc[i - 1, "USD Traded"]
            )

    if df.loc[i, "Signal"] == 2:
        df.loc[i, "USD Traded"] = df.loc[i - 2, "USD Traded"]
        df.loc[i, "% Traded"] = df.loc[i - 2, "% Traded"]
        if df.loc[i - 2, "ratio eq"] > 0.6:
            df.loc[i, "EQ USD"] = (
                df.loc[i - 1, "EQ USD"] * df.loc[i, "eq_return_+1"]
                - 1 / 2 * df.loc[i - 2, "USD Traded"]
            )
            df.loc[i, "T USD"] = (
                df.loc[i - 1, "T USD"] * df.loc[i, "t_return_+1"]
                + 1 / 2 * df.loc[i - 2, "USD Traded"]
            )
        if df.loc[i - 2, "ratio t"] > 0.4:
            df.loc[i, "EQ USD"] = (
                df.loc[i - 1, "EQ USD"] * df.loc[i, "eq_return_+1"]
                + 1 / 2 * df.loc[i - 2, "USD Traded"]
            )
            df.loc[i, "T USD"] = (
                df.loc[i - 1, "T USD"] * df.loc[i, "t_return_+1"]
                - 1 / 2 * df.loc[i - 2, "USD Traded"]
            )


# Combined return
for i in range(1, len(df)):
    df.loc[i, "total_return"] = (
        df.loc[i - 1, "ratio eq"] * df.loc[i, "eq_return_+1"]
        + df.loc[i - 1, "ratio t"] * df.loc[i, "t_return_+1"]
    )


# Combined return - transk
for i in range(1, len(df)):
    if df.loc[i, "Signal"] == 0:
        df.loc[i, "total_return-tk"] = (
            df.loc[i - 1, "ratio eq"] * df.loc[i, "eq_return_+1"]
            + df.loc[i - 1, "ratio t"] * df.loc[i, "t_return_+1"]
        )
    if df.loc[i, "Signal"] != 0:
        df.loc[i, "total_return-tk"] = (
            df.loc[i - 1, "ratio eq"] * df.loc[i, "eq_return_+1"]
            + df.loc[i - 1, "ratio t"] * df.loc[i, "t_return_+1"]
            - 1 / 2 * df.loc[i, "% Traded"] * 0.015
        )


# Geo mittel
y = 1
for i in range(1, len(df)):
    y = y * df.loc[i, "total_return"]

geom = y ** (1 / len(df)) - 1
print(geom)

# Geo mittel trans
z = 1
for i in range(1, len(df)):
    z = z * df.loc[i, "total_return-tk"]

geomtk = z ** (1 / len(df)) - 1
print(geomtk)

# Speichern des Zwischenstandes
df.to_excel(
    r"C:\Users\Mayer\Desktop\Data Science Ohkrin\python\Daten\2 Tage linear.xlsx"
)

# Linear 3 day
df = df_main.copy()

# Signal
df["Signal"] = 0
df.iloc[::30]["Signal"] = 1

for i in range(1, len(df)):
    if df.loc[i, "Signal"] == 1:
        df.loc[i + 1, "Signal"] = 2
        df.loc[i + 2, "Signal"] = 3

# Hilfsvariablen
df["EQ USD"] = 0
df["T USD"] = 0
df["USD ALL"] = 0
df["USD Traded"] = 0
df["% Traded"] = 0
df["total_return"] = 0
df["total_return-tk"] = 0
df.loc[0, "EQ USD"] = 0.6
df.loc[0, "T USD"] = 0.4
df = df.fillna(method="bfill")
df = df.drop(
    ["EQ OPEN", "EQ HIGH", "EQ LOW", "T OPEN", "T HIGH", "T LOW", "Währung"], axis=1
)
df["eq_return"] = df["EQ CLOSE"].pct_change()
df["t_return"] = df["T CLOSE"].pct_change()
df["eq_return_+1"] = df["eq_return"] + 1
df["t_return_+1"] = df["t_return"] + 1
df["eq_return_+1"].fillna(1, inplace=True)
df["t_return_+1"].fillna(1, inplace=True)

# investierte USD
for i in range(1, len(df)):
    if df.loc[i, "Signal"] == 0:
        df.loc[i, "EQ USD"] = df.loc[i - 1, "EQ USD"] * df.loc[i, "eq_return_+1"]

        df.loc[i, "T USD"] = df.loc[i - 1, "T USD"] * df.loc[i, "t_return_+1"]

        df["USD ALL"] = df["EQ USD"] + df["T USD"]
        df["ratio eq"] = df["EQ USD"] / (df["USD ALL"])
        df["ratio t"] = df["T USD"] / df["USD ALL"]

        if df.loc[i, "ratio eq"] > 0.6:
            df.loc[i, "USD Traded"] = df.loc[i, "EQ USD"] - 0.6 * df.loc[i, "USD ALL"]

        if df.loc[i, "ratio t"] > 0.4:
            df.loc[i, "USD Traded"] = df.loc[i, "T USD"] - 0.4 * df.loc[i, "USD ALL"]

        df.loc[i, "% Traded"] = df.loc[i, "USD Traded"] / df.loc[i, "USD ALL"]

    if df.loc[i, "Signal"] == 1:
        df.loc[i, "USD Traded"] = df.loc[i - 1, "USD Traded"]
        df.loc[i, "% Traded"] = df.loc[i - 1, "% Traded"]
        if df.loc[i - 1, "ratio eq"] > 0.6:
            df.loc[i, "EQ USD"] = (
                df.loc[i - 1, "EQ USD"] * df.loc[i, "eq_return_+1"]
                - 1 / 3 * df.loc[i - 1, "USD Traded"]
            )
            df.loc[i, "T USD"] = (
                df.loc[i - 1, "T USD"] * df.loc[i, "t_return_+1"]
                + 1 / 3 * df.loc[i - 1, "USD Traded"]
            )
        if df.loc[i - 1, "ratio t"] > 0.4:
            df.loc[i, "EQ USD"] = (
                df.loc[i - 1, "EQ USD"] * df.loc[i, "eq_return_+1"]
                + 1 / 3 * df.loc[i - 1, "USD Traded"]
            )
            df.loc[i, "T USD"] = (
                df.loc[i - 1, "T USD"] * df.loc[i, "t_return_+1"]
                - 1 / 3 * df.loc[i - 1, "USD Traded"]
            )

    if df.loc[i, "Signal"] == 2:
        df.loc[i, "USD Traded"] = df.loc[i - 2, "USD Traded"]
        df.loc[i, "% Traded"] = df.loc[i - 2, "% Traded"]
        if df.loc[i - 2, "ratio eq"] > 0.6:
            df.loc[i, "EQ USD"] = (
                df.loc[i - 1, "EQ USD"] * df.loc[i, "eq_return_+1"]
                - 1 / 3 * df.loc[i - 2, "USD Traded"]
            )
            df.loc[i, "T USD"] = (
                df.loc[i - 1, "T USD"] * df.loc[i, "t_return_+1"]
                + 1 / 3 * df.loc[i - 2, "USD Traded"]
            )
        if df.loc[i - 2, "ratio t"] > 0.4:
            df.loc[i, "EQ USD"] = (
                df.loc[i - 1, "EQ USD"] * df.loc[i, "eq_return_+1"]
                + 1 / 3 * df.loc[i - 2, "USD Traded"]
            )
            df.loc[i, "T USD"] = (
                df.loc[i - 1, "T USD"] * df.loc[i, "t_return_+1"]
                - 1 / 3 * df.loc[i - 2, "USD Traded"]
            )

    if df.loc[i, "Signal"] == 3:
        df.loc[i, "USD Traded"] = df.loc[i - 3, "USD Traded"]
        df.loc[i, "% Traded"] = df.loc[i - 3, "% Traded"]
        if df.loc[i - 3, "ratio eq"] > 0.6:
            df.loc[i, "EQ USD"] = (
                df.loc[i - 1, "EQ USD"] * df.loc[i, "eq_return_+1"]
                - 1 / 3 * df.loc[i - 3, "USD Traded"]
            )
            df.loc[i, "T USD"] = (
                df.loc[i - 1, "T USD"] * df.loc[i, "t_return_+1"]
                + 1 / 3 * df.loc[i - 3, "USD Traded"]
            )
        if df.loc[i - 3, "ratio t"] > 0.4:
            df.loc[i, "EQ USD"] = (
                df.loc[i - 1, "EQ USD"] * df.loc[i, "eq_return_+1"]
                + 1 / 3 * df.loc[i - 3, "USD Traded"]
            )
            df.loc[i, "T USD"] = (
                df.loc[i - 1, "T USD"] * df.loc[i, "t_return_+1"]
                - 1 / 3 * df.loc[i - 3, "USD Traded"]
            )


# Combined return
for i in range(1, len(df)):
    df.loc[i, "total_return"] = (
        df.loc[i - 1, "ratio eq"] * df.loc[i, "eq_return_+1"]
        + df.loc[i - 1, "ratio t"] * df.loc[i, "t_return_+1"]
    )


# Combined return - transk
for i in range(1, len(df)):
    if df.loc[i, "Signal"] == 0:
        df.loc[i, "total_return-tk"] = (
            df.loc[i - 1, "ratio eq"] * df.loc[i, "eq_return_+1"]
            + df.loc[i - 1, "ratio t"] * df.loc[i, "t_return_+1"]
        )
    if df.loc[i, "Signal"] != 0:
        df.loc[i, "total_return-tk"] = (
            df.loc[i - 1, "ratio eq"] * df.loc[i, "eq_return_+1"]
            + df.loc[i - 1, "ratio t"] * df.loc[i, "t_return_+1"]
            - 1 / 3 * df.loc[i, "% Traded"] * 0.015
        )


# geo mittel
y = 1
for i in range(1, len(df)):
    y = y * df.loc[i, "total_return"]

geom = y ** (1 / len(df)) - 1
print(geom)

# Speichern des Zwischenstandes
df.to_excel(
    r"C:\Users\Mayer\Desktop\Data Science Ohkrin\python\Daten\3 Tage linear.xlsx"
)

# Progressive 2 day
df = df_main.copy()

# Signale
df["Signal"] = 0
df.iloc[::30]["Signal"] = 1

for i in range(1, len(df)):
    if df.loc[i, "Signal"] == 1:
        df.loc[i + 1, "Signal"] = 2

# Hilfsvariablen
df["EQ USD"] = 0
df["T USD"] = 0
df["USD ALL"] = 0
df["USD Traded"] = 0
df["% Traded"] = 0
df["total_return"] = 0
df["total_return-tk"] = 0
df.loc[0, "EQ USD"] = 0.6
df.loc[0, "T USD"] = 0.4
df = df.fillna(method="bfill")
df = df.drop(
    ["EQ OPEN", "EQ HIGH", "EQ LOW", "T OPEN", "T HIGH", "T LOW", "Währung"], axis=1
)
df["eq_return"] = df["EQ CLOSE"].pct_change()
df["t_return"] = df["T CLOSE"].pct_change()
df["eq_return_+1"] = df["eq_return"] + 1
df["t_return_+1"] = df["t_return"] + 1
df["eq_return_+1"].fillna(1, inplace=True)
df["t_return_+1"].fillna(1, inplace=True)

# investierte USD
for i in range(1, len(df)):
    if df.loc[i, "Signal"] == 0:
        df.loc[i, "EQ USD"] = df.loc[i - 1, "EQ USD"] * df.loc[i, "eq_return_+1"]

        df.loc[i, "T USD"] = df.loc[i - 1, "T USD"] * df.loc[i, "t_return_+1"]

        df["USD ALL"] = df["EQ USD"] + df["T USD"]
        df["ratio eq"] = df["EQ USD"] / (df["USD ALL"])
        df["ratio t"] = df["T USD"] / df["USD ALL"]

        if df.loc[i, "ratio eq"] > 0.6:
            df.loc[i, "USD Traded"] = df.loc[i, "EQ USD"] - 0.6 * df.loc[i, "USD ALL"]

        if df.loc[i, "ratio t"] > 0.4:
            df.loc[i, "USD Traded"] = df.loc[i, "T USD"] - 0.4 * df.loc[i, "USD ALL"]

        df.loc[i, "% Traded"] = df.loc[i, "USD Traded"] / df.loc[i, "USD ALL"]

    if df.loc[i, "Signal"] == 1:
        df.loc[i, "USD Traded"] = df.loc[i - 1, "USD Traded"]
        df.loc[i, "% Traded"] = df.loc[i - 1, "% Traded"]
        if df.loc[i - 1, "ratio eq"] > 0.6:
            df.loc[i, "EQ USD"] = (
                df.loc[i - 1, "EQ USD"] * df.loc[i, "eq_return_+1"]
                - 6 / 10 * df.loc[i - 1, "USD Traded"]
            )
            df.loc[i, "T USD"] = (
                df.loc[i - 1, "T USD"] * df.loc[i, "t_return_+1"]
                + 6 / 10 * df.loc[i - 1, "USD Traded"]
            )
        if df.loc[i - 1, "ratio t"] > 0.4:
            df.loc[i, "EQ USD"] = (
                df.loc[i - 1, "EQ USD"] * df.loc[i, "eq_return_+1"]
                + 6 / 10 * df.loc[i - 1, "USD Traded"]
            )
            df.loc[i, "T USD"] = (
                df.loc[i - 1, "T USD"] * df.loc[i, "t_return_+1"]
                - 6 / 10 * df.loc[i - 1, "USD Traded"]
            )

    if df.loc[i, "Signal"] == 2:
        df.loc[i, "USD Traded"] = df.loc[i - 2, "USD Traded"]
        df.loc[i, "% Traded"] = df.loc[i - 2, "% Traded"]
        if df.loc[i - 2, "ratio eq"] > 0.6:
            df.loc[i, "EQ USD"] = (
                df.loc[i - 1, "EQ USD"] * df.loc[i, "eq_return_+1"]
                - 4 / 10 * df.loc[i - 2, "USD Traded"]
            )
            df.loc[i, "T USD"] = (
                df.loc[i - 1, "T USD"] * df.loc[i, "t_return_+1"]
                + 4 / 10 * df.loc[i - 2, "USD Traded"]
            )
        if df.loc[i - 2, "ratio t"] > 0.4:
            df.loc[i, "EQ USD"] = (
                df.loc[i - 1, "EQ USD"] * df.loc[i, "eq_return_+1"]
                + 4 / 10 * df.loc[i - 2, "USD Traded"]
            )
            df.loc[i, "T USD"] = (
                df.loc[i - 1, "T USD"] * df.loc[i, "t_return_+1"]
                - 4 / 10 * df.loc[i - 2, "USD Traded"]
            )


# Combined return
for i in range(1, len(df)):
    df.loc[i, "total_return"] = (
        df.loc[i - 1, "ratio eq"] * df.loc[i, "eq_return_+1"]
        + df.loc[i - 1, "ratio t"] * df.loc[i, "t_return_+1"]
    )

# Combined return - transk
for i in range(1, len(df)):
    if df.loc[i, "Signal"] == 0:
        df.loc[i, "total_return-tk"] = (
            df.loc[i - 1, "ratio eq"] * df.loc[i, "eq_return_+1"]
            + df.loc[i - 1, "ratio t"] * df.loc[i, "t_return_+1"]
        )
    if df.loc[i, "Signal"] == 1:
        df.loc[i, "total_return-tk"] = (
            df.loc[i - 1, "ratio eq"] * df.loc[i, "eq_return_+1"]
            + df.loc[i - 1, "ratio t"] * df.loc[i, "t_return_+1"]
            - 6 / 10 * df.loc[i, "% Traded"] * 0.015
        )
    if df.loc[i, "Signal"] == 2:
        df.loc[i, "total_return-tk"] = (
            df.loc[i - 1, "ratio eq"] * df.loc[i, "eq_return_+1"]
            + df.loc[i - 1, "ratio t"] * df.loc[i, "t_return_+1"]
            - 4 / 10 * df.loc[i, "% Traded"] * 0.015
        )

# geo mittel
y = 1
for i in range(1, len(df)):
    y = y * df.loc[i, "total_return"]


geom = y ** (1 / len(df)) - 1
print(geom)

df.to_excel(
    r"C:\Users\Mayer\Desktop\Data Science Ohkrin\python\Daten\2 Tage progressiv.xlsx"
)

# Progressive 3 day
df = df_main.copy()

# Signale
df["Signal"] = 0
df.iloc[::30]["Signal"] = 1

for i in range(1, len(df)):
    if df.loc[i, "Signal"] == 1:
        df.loc[i + 1, "Signal"] = 2
        df.loc[i + 2, "Signal"] = 3

# Hilfsvariablen
df["EQ USD"] = 0
df["T USD"] = 0
df["USD ALL"] = 0
df["USD Traded"] = 0
df["% Traded"] = 0
df["total_return"] = 0
df["total_return-tk"] = 0
df.loc[0, "EQ USD"] = 0.6
df.loc[0, "T USD"] = 0.4
df = df.fillna(method="bfill")
df = df.drop(
    ["EQ OPEN", "EQ HIGH", "EQ LOW", "T OPEN", "T HIGH", "T LOW", "Währung"], axis=1
)
df["eq_return"] = df["EQ CLOSE"].pct_change()
df["t_return"] = df["T CLOSE"].pct_change()
df["eq_return_+1"] = df["eq_return"] + 1
df["t_return_+1"] = df["t_return"] + 1
df["eq_return_+1"].fillna(1, inplace=True)
df["t_return_+1"].fillna(1, inplace=True)

# Investierte USD
for i in range(1, len(df)):
    if df.loc[i, "Signal"] == 0:
        df.loc[i, "EQ USD"] = df.loc[i - 1, "EQ USD"] * df.loc[i, "eq_return_+1"]

        df.loc[i, "T USD"] = df.loc[i - 1, "T USD"] * df.loc[i, "t_return_+1"]

        df["USD ALL"] = df["EQ USD"] + df["T USD"]
        df["ratio eq"] = df["EQ USD"] / (df["USD ALL"])
        df["ratio t"] = df["T USD"] / df["USD ALL"]

        if df.loc[i, "ratio eq"] > 0.6:
            df.loc[i, "USD Traded"] = df.loc[i, "EQ USD"] - 0.6 * df.loc[i, "USD ALL"]

        if df.loc[i, "ratio t"] > 0.4:
            df.loc[i, "USD Traded"] = df.loc[i, "T USD"] - 0.4 * df.loc[i, "USD ALL"]

        df.loc[i, "% Traded"] = df.loc[i, "USD Traded"] / df.loc[i, "USD ALL"]

    if df.loc[i, "Signal"] == 1:
        df.loc[i, "USD Traded"] = df.loc[i - 1, "USD Traded"]
        df.loc[i, "% Traded"] = df.loc[i - 1, "% Traded"]
        if df.loc[i - 1, "ratio eq"] > 0.6:
            df.loc[i, "EQ USD"] = (
                df.loc[i - 1, "EQ USD"] * df.loc[i, "eq_return_+1"]
                - 1 / 2 * df.loc[i - 1, "USD Traded"]
            )
            df.loc[i, "T USD"] = (
                df.loc[i - 1, "T USD"] * df.loc[i, "t_return_+1"]
                + 1 / 2 * df.loc[i - 1, "USD Traded"]
            )
        if df.loc[i - 1, "ratio t"] > 0.4:
            df.loc[i, "EQ USD"] = (
                df.loc[i - 1, "EQ USD"] * df.loc[i, "eq_return_+1"]
                + 1 / 2 * df.loc[i - 1, "USD Traded"]
            )
            df.loc[i, "T USD"] = (
                df.loc[i - 1, "T USD"] * df.loc[i, "t_return_+1"]
                - 1 / 2 * df.loc[i - 1, "USD Traded"]
            )

    if df.loc[i, "Signal"] == 2:
        df.loc[i, "USD Traded"] = df.loc[i - 2, "USD Traded"]
        df.loc[i, "% Traded"] = df.loc[i - 2, "% Traded"]
        if df.loc[i - 2, "ratio eq"] > 0.6:
            df.loc[i, "EQ USD"] = (
                df.loc[i - 1, "EQ USD"] * df.loc[i, "eq_return_+1"]
                - 3 / 10 * df.loc[i - 2, "USD Traded"]
            )
            df.loc[i, "T USD"] = (
                df.loc[i - 1, "T USD"] * df.loc[i, "t_return_+1"]
                + 3 / 10 * df.loc[i - 2, "USD Traded"]
            )
        if df.loc[i - 2, "ratio t"] > 0.4:
            df.loc[i, "EQ USD"] = (
                df.loc[i - 1, "EQ USD"] * df.loc[i, "eq_return_+1"]
                + 3 / 10 * df.loc[i - 2, "USD Traded"]
            )
            df.loc[i, "T USD"] = (
                df.loc[i - 1, "T USD"] * df.loc[i, "t_return_+1"]
                - 3 / 10 * df.loc[i - 2, "USD Traded"]
            )

    if df.loc[i, "Signal"] == 3:
        df.loc[i, "USD Traded"] = df.loc[i - 3, "USD Traded"]
        df.loc[i, "% Traded"] = df.loc[i - 3, "% Traded"]
        if df.loc[i - 3, "ratio eq"] > 0.6:
            df.loc[i, "EQ USD"] = (
                df.loc[i - 1, "EQ USD"] * df.loc[i, "eq_return_+1"]
                - 1 / 5 * df.loc[i - 3, "USD Traded"]
            )
            df.loc[i, "T USD"] = (
                df.loc[i - 1, "T USD"] * df.loc[i, "t_return_+1"]
                + 1 / 5 * df.loc[i - 3, "USD Traded"]
            )
        if df.loc[i - 3, "ratio t"] > 0.4:
            df.loc[i, "EQ USD"] = (
                df.loc[i - 1, "EQ USD"] * df.loc[i, "eq_return_+1"]
                + 1 / 5 * df.loc[i - 3, "USD Traded"]
            )
            df.loc[i, "T USD"] = (
                df.loc[i - 1, "T USD"] * df.loc[i, "t_return_+1"]
                - 1 / 5 * df.loc[i - 3, "USD Traded"]
            )

# Combined return
for i in range(1, len(df)):
    df.loc[i, "total_return"] = (
        df.loc[i - 1, "ratio eq"] * df.loc[i, "eq_return_+1"]
        + df.loc[i - 1, "ratio t"] * df.loc[i, "t_return_+1"]
    )


# Combined return - transk
for i in range(1, len(df)):
    if df.loc[i, "Signal"] == 0:
        df.loc[i, "total_return-tk"] = (
            df.loc[i - 1, "ratio eq"] * df.loc[i, "eq_return_+1"]
            + df.loc[i - 1, "ratio t"] * df.loc[i, "t_return_+1"]
        )
    if df.loc[i, "Signal"] == 1:
        df.loc[i, "total_return-tk"] = (
            df.loc[i - 1, "ratio eq"] * df.loc[i, "eq_return_+1"]
            + df.loc[i - 1, "ratio t"] * df.loc[i, "t_return_+1"]
            - 1 / 2 * df.loc[i, "% Traded"] * 0.015
        )
    if df.loc[i, "Signal"] == 2:
        df.loc[i, "total_return-tk"] = (
            df.loc[i - 1, "ratio eq"] * df.loc[i, "eq_return_+1"]
            + df.loc[i - 1, "ratio t"] * df.loc[i, "t_return_+1"]
            - 3 / 10 * df.loc[i, "% Traded"] * 0.015
        )
    if df.loc[i, "Signal"] == 3:
        df.loc[i, "total_return-tk"] = (
            df.loc[i - 1, "ratio eq"] * df.loc[i, "eq_return_+1"]
            + df.loc[i - 1, "ratio t"] * df.loc[i, "t_return_+1"]
            - 1 / 5 * df.loc[i, "% Traded"] * 0.015
        )


# Geo mittel
y = 1
for i in range(1, len(df)):
    y = y * df.loc[i, "total_return"]
geom = y ** (1 / len(df)) - 1
print(geom)

# Speichern des Zwischenstandes
df.to_excel(
    r"C:\Users\Mayer\Desktop\Data Science Ohkrin\python\Daten\3 Tage progressiv.xlsx"
)
