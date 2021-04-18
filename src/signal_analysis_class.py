# Data Decision Science

"""
Modul zur Timimng-Analyse hinsichtlich der Umsetzung von verschiedenen 
technischen Anlagestrategien 
"""

# Eigene Module
import main_functions as mf

# Allgemeine Module
import pandas as pd
import os
import numpy as np
from scipy.stats.mstats import gmean

# Klasse
class signals_analysis:
    """Klassenmodul, anhand dessen die verwendeten Crossover- und Countertrend-Strategien automatisiert untersucht werden können."""

    # Allgemein
    # Instantiation
    def __init__(self, filename, wd_main, wd_market_data, wd_other_data, wd_export):
        self.filename = filename
        self.wd_main = wd_main
        self.wd_market_data = wd_market_data
        self.wd_other_data = wd_other_data
        self.wd_export = wd_export

        # Grundlegende Informationen
        # Spaltennamen der Rohdaten
        self.high_column = "px_high"
        self.low_column = "px_low"
        self.close_column = "px_last"
        self.open_column = "px_open"
        self.currency_column = "currency"
        self.crossover_column = "crossover_signal"
        self.countertrend_column = "countertrend_signal"

        # Working Directories und Dateien
        self.risk_premiums_file = "Risk_Premium.csv"
        self.risk_premium_column = "Mkt-RF"
        self.export_kind = "full_transition"

        # Methodische Einstellungen
        # Allgmein
        self.transaction_costs = 0.005  # Prozentuale Transaktionskosten
        self.return_kind = "normal"  # Renditenart
        self.transitions = {"days": [1, 2, 3], "portions": [0.33, 0.67, 1]}
        # Bootstraps
        self.bootstraps_replicates = 0  # Anzahl der Bootstraps
        self.chunksize = 180  # Chunksize der Bootstraps
        self.seed = 123  # Seed der Bootstraps
        # Crossover
        self.co_avg_short = 38  # kurzer Zeitraum der gleitenden Durchschnitte
        self.co_avg_long = 200  # langer Zeitraum der gleitenden Durchschnitte
        # Countertrend
        self.std_window = 7  # Zeitfenster der Laufenden Standardabweichung, dient der Simulation der High- und Low-Series
        self.trend = "ma"  # Methode zu Berechnung des Trends, entweder Moving Average ('ma') oder EWM ('ewm')
        self.ct_avg_short = 38  # Kurzer Zeitraum der Trendberechnung
        self.ct_avg_long = 200  # Lager Zeitraum der Trendberechnung
        self.window_reading = 20  # Zeitraum der Readings
        self.true_range = 20  # True Range des Plungers (oftmals auch 14 Tage)
        self.plunge_trigger = 3  # Schwelle, an der der Plunger getriggert wird
        self.confirmation_period = (
            2  # Confirmation Period (lediglich bezüglich des Exits!)
        )
        # Z-Scores
        self.zscore_window = 180  # Zeitfenster für die laufenden Z-Scores
        self.zscore_period = 50  # Investitionszeitraum der Mean Reversion basierend auf den laufenden Z-Scores

    # Empirie
    def main_benchmark(self):
        """Funktion, die die jeweiligen Benchmark-Ergebnisse errechnet und in dementsprechende Excel-Dateien ab-
        speichert."""
        # Einlesen der DataFrames und Series
        os.chdir(self.wd_market_data)
        df_market = pd.read_csv(
            self.filename,
            header=0,
            index_col="date",
            parse_dates=["date"],
            dayfirst=True,
        )
        os.chdir(self.wd_other_data)
        risk_premium_rough = pd.read_csv(
            self.risk_premiums_file,
            header=0,
            index_col="date",
            parse_dates=["date"],
            dayfirst=True,
        )
        risk_premium = df_market.merge(risk_premium_rough, how="left", on="date")[
            self.risk_premium_column
        ]

        # Benchmark ohne Bootstraps
        bench_wo_boot = {}
        bench_return_rates = {}
        # Buy-and-Hold
        buy_and_hold_returns = mf.rate_of_return(
            df_market[self.close_column], kind=self.return_kind
        )
        bench_return_rates["buy_and_hold"] = buy_and_hold_returns
        buy_and_hold_sr = mf.sharpe_ratio(buy_and_hold_returns)
        buy_and_hold_alpha = mf.calc_alpha(buy_and_hold_returns, risk_premium)
        buy_and_hold_std = np.std(buy_and_hold_returns)
        buy_and_hold_mean = np.mean(buy_and_hold_returns)
        buy_and_hold_gmean = gmean((buy_and_hold_returns + 1).fillna(1)) - 1
        buy_and_hold_median = np.percentile(buy_and_hold_returns, 50)
        buy_and_hold_25p = np.percentile(buy_and_hold_returns, 25)
        buy_and_hold_75p = np.percentile(buy_and_hold_returns, 75)
        bench_wo_boot["buy_and_hold"] = [
            buy_and_hold_sr,
            buy_and_hold_alpha,
            buy_and_hold_std,
            buy_and_hold_mean,
            buy_and_hold_gmean,
            buy_and_hold_median,
            buy_and_hold_25p,
            buy_and_hold_75p,
        ]

        # Crossover gemäß Transition
        signals = df_market[self.crossover_column]
        co_wo_cost_returns, co_w_cost_returns = mf.returns_weights(
            signals,
            buy_and_hold_returns,
            self.transitions,
            kind=self.return_kind,
            transaction_costs=self.transaction_costs,
        )
        bench_return_rates["co_wo_cost_returns"] = co_wo_cost_returns
        bench_return_rates["co_w_cost_returns"] = co_w_cost_returns

        #% Ohne Transaktionskosten
        co_wo_cost_sr = mf.sharpe_ratio(co_wo_cost_returns)
        co_wo_cost_alpha = mf.calc_alpha(co_wo_cost_returns, risk_premium)
        co_wo_cost_std = np.std(co_wo_cost_returns)
        co_wo_cost_mean = np.mean(co_wo_cost_returns)
        co_wo_cost_gmean = gmean((co_wo_cost_returns + 1).fillna(1)) - 1
        co_wo_cost_median = np.percentile(co_wo_cost_returns, 50)
        co_wo_cost_25p = np.percentile(co_wo_cost_returns, 25)
        co_wo_cost_75p = np.percentile(co_wo_cost_returns, 75)
        bench_wo_boot["crossover_wo_costs"] = [
            co_wo_cost_sr,
            co_wo_cost_alpha,
            co_wo_cost_std,
            co_wo_cost_mean,
            co_wo_cost_gmean,
            co_wo_cost_median,
            co_wo_cost_25p,
            co_wo_cost_75p,
        ]

        #     #% Mit Transaktionskosten
        co_w_cost_sr = mf.sharpe_ratio(co_w_cost_returns)
        co_w_cost_alpha = mf.calc_alpha(co_w_cost_returns, risk_premium)
        co_w_cost_std = np.std(co_w_cost_returns)
        co_w_cost_mean = np.mean(co_w_cost_returns)
        co_w_cost_gmean = gmean((co_w_cost_returns + 1).fillna(1)) - 1
        co_w_cost_median = np.percentile(co_w_cost_returns, 50)
        co_w_cost_25p = np.percentile(co_w_cost_returns, 25)
        co_w_cost_75p = np.percentile(co_w_cost_returns, 75)
        bench_wo_boot["crossover_w_costs"] = [
            co_w_cost_sr,
            co_w_cost_alpha,
            co_w_cost_std,
            co_w_cost_mean,
            co_w_cost_gmean,
            co_w_cost_median,
            co_w_cost_25p,
            co_w_cost_75p,
        ]

        #% Transition Countertrend
        signals = df_market[self.countertrend_column]
        ct_wo_cost_returns, ct_w_cost_returns = mf.returns_weights(
            signals,
            buy_and_hold_returns,
            self.transitions,
            kind=self.return_kind,
            transaction_costs=self.transaction_costs,
        )
        bench_return_rates["ct_wo_cost_returns"] = ct_wo_cost_returns
        bench_return_rates["ct_w_cost_returns"] = ct_w_cost_returns

        #% Ohne Transaktionskosten
        ct_wo_cost_sr = mf.sharpe_ratio(ct_wo_cost_returns)
        ct_wo_cost_alpha = mf.calc_alpha(ct_wo_cost_returns, risk_premium)
        ct_wo_cost_std = np.std(ct_wo_cost_returns)
        ct_wo_cost_mean = np.mean(ct_wo_cost_returns)
        ct_wo_cost_gmean = gmean((ct_wo_cost_returns + 1).fillna(1)) - 1
        ct_wo_cost_median = np.percentile(ct_wo_cost_returns, 50)
        ct_wo_cost_25p = np.percentile(ct_wo_cost_returns, 25)
        ct_wo_cost_75p = np.percentile(ct_wo_cost_returns, 75)
        bench_wo_boot["countertrend_wo_costs"] = [
            ct_wo_cost_sr,
            ct_wo_cost_alpha,
            ct_wo_cost_std,
            ct_wo_cost_mean,
            ct_wo_cost_gmean,
            ct_wo_cost_median,
            ct_wo_cost_25p,
            ct_wo_cost_75p,
        ]

        #     #% Mit Transaktionskosten
        ct_w_cost_sr = mf.sharpe_ratio(ct_w_cost_returns)
        ct_w_cost_alpha = mf.calc_alpha(ct_w_cost_returns, risk_premium)
        ct_w_cost_std = np.std(ct_w_cost_returns)
        ct_w_cost_mean = np.mean(ct_w_cost_returns)
        ct_w_cost_gmean = gmean((ct_w_cost_returns + 1).fillna(1)) - 1
        ct_w_cost_median = np.percentile(ct_w_cost_returns, 50)
        ct_w_cost_25p = np.percentile(ct_w_cost_returns, 25)
        ct_w_cost_75p = np.percentile(ct_w_cost_returns, 75)
        bench_wo_boot["countertrend_w_costs"] = [
            ct_w_cost_sr,
            ct_w_cost_alpha,
            ct_w_cost_std,
            ct_w_cost_mean,
            ct_w_cost_gmean,
            ct_w_cost_median,
            ct_w_cost_25p,
            ct_w_cost_75p,
        ]
        #% Transition z-scores
        signals = mf.z_score_signals(
            df_market[self.close_column],
            window=self.zscore_window,
            period=self.zscore_period,
        )
        zs_wo_cost_returns, zs_w_cost_returns = mf.returns_weights(
            signals,
            buy_and_hold_returns,
            self.transitions,
            kind=self.return_kind,
            transaction_costs=self.transaction_costs,
        )
        bench_return_rates["zs_wo_cost_returns"] = zs_wo_cost_returns
        bench_return_rates["zs_w_cost_returns"] = zs_w_cost_returns

        #% Ohne Transaktionskosten
        zs_wo_cost_sr = mf.sharpe_ratio(zs_wo_cost_returns)
        zs_wo_cost_alpha = mf.calc_alpha(zs_wo_cost_returns, risk_premium)
        zs_wo_cost_std = np.std(zs_wo_cost_returns)
        zs_wo_cost_mean = np.mean(zs_wo_cost_returns)
        zs_wo_cost_gmean = gmean((zs_wo_cost_returns + 1).fillna(1)) - 1
        zs_wo_cost_median = np.percentile(zs_wo_cost_returns, 50)
        zs_wo_cost_25p = np.percentile(zs_wo_cost_returns, 25)
        zs_wo_cost_75p = np.percentile(zs_wo_cost_returns, 75)
        bench_wo_boot["zscore_wo_costs"] = [
            zs_wo_cost_sr,
            zs_wo_cost_alpha,
            zs_wo_cost_std,
            zs_wo_cost_mean,
            zs_wo_cost_gmean,
            zs_wo_cost_median,
            zs_wo_cost_25p,
            zs_wo_cost_75p,
        ]

        #% Mit Transaktionskosten
        zs_w_cost_sr = mf.sharpe_ratio(zs_w_cost_returns)
        zs_w_cost_alpha = mf.calc_alpha(zs_w_cost_returns, risk_premium)
        zs_w_cost_std = np.std(zs_w_cost_returns)
        zs_w_cost_mean = np.mean(zs_w_cost_returns)
        zs_w_cost_gmean = gmean((zs_w_cost_returns + 1).fillna(1)) - 1
        zs_w_cost_median = np.percentile(zs_w_cost_returns, 50)
        zs_w_cost_25p = np.percentile(zs_w_cost_returns, 25)
        zs_w_cost_75p = np.percentile(zs_w_cost_returns, 75)
        bench_wo_boot["zscore_w_costs"] = [
            zs_w_cost_sr,
            zs_w_cost_alpha,
            zs_w_cost_std,
            zs_w_cost_mean,
            zs_w_cost_gmean,
            zs_w_cost_median,
            zs_w_cost_25p,
            zs_w_cost_75p,
        ]

        #% Zusammenführen der Ergebnisse
        df_bench = pd.DataFrame(bench_wo_boot)
        df_bench.index = [
            "sharpe_ratio",
            "alpha",
            "std",
            "mean",
            "gmean",
            "median",
            "25p",
            "75p",
        ]
        df_returns_bench = pd.DataFrame(bench_return_rates)

        #% Abspeichern der Ergebnisse
        os.chdir(self.wd_export)
        export_filename = self.filename[:3] + "_benchmark_" + self.export_kind + ".xlsx"
        returns_filename = (
            self.filename[:3] + "_return_rates_benchmark_" + self.export_kind + ".csv"
        )
        df_returns_bench.to_csv(returns_filename, header=True)
        df_bench.to_excel(export_filename)

    def main_bootstraps(self):
        """Funktion, die die jeweiligen Ergebnisse der Bootstraps ausgibt und in CSV-Dateien abspeichert."""
        if self.bootstraps_replicates == 0:
            bootstraps_results = pd.DataFrame()
            print("Bootstraps-Anzahl ist 0!")
        else:
            #% Einlesen der DataFrames und Series
            os.chdir(self.wd_market_data)
            df_market = pd.read_csv(
                self.filename,
                header=0,
                index_col="date",
                parse_dates=["date"],
                dayfirst=True,
            )
            os.chdir(self.wd_other_data)
            risk_premium_rough = pd.read_csv(
                self.risk_premiums_file,
                header=0,
                index_col="date",
                parse_dates=["date"],
                dayfirst=True,
            )
            risk_premium = df_market.merge(risk_premium_rough, how="left", on="date")[
                self.risk_premium_column
            ]

            #% Ursprungsrendite
            original_returns = mf.rate_of_return(
                df_market[self.close_column], kind=self.return_kind
            )

            #% Bootstraps
            bootstraps = mf.bootstrap(
                df_market[self.close_column],
                original_returns,
                number_replicates=self.bootstraps_replicates,
                chunksize=self.chunksize,
                seed=self.seed,
            )

            #% Spalten
            #% Buy-and-Hold
            buy_and_hold_sr = []
            buy_and_hold_alpha = []
            buy_and_hold_std = []
            buy_and_hold_mean = []
            buy_and_hold_gmean = []
            buy_and_hold_median = []
            buy_and_hold_25p = []
            buy_and_hold_75p = []
            #% Crossover
            #% Ohne Transaktionskosten
            co_wo_cost_sr = []
            co_wo_cost_alpha = []
            co_wo_cost_std = []
            co_wo_cost_mean = []
            co_wo_cost_gmean = []
            co_wo_cost_median = []
            co_wo_cost_25p = []
            co_wo_cost_75p = []
            #% Ohne Transaktionskosten
            co_w_cost_sr = []
            co_w_cost_alpha = []
            co_w_cost_std = []
            co_w_cost_mean = []
            co_w_cost_gmean = []
            co_w_cost_median = []
            co_w_cost_25p = []
            co_w_cost_75p = []
            #% Countertrend
            ct_wo_cost_sr = []
            ct_wo_cost_alpha = []
            ct_wo_cost_std = []
            ct_wo_cost_mean = []
            ct_wo_cost_gmean = []
            ct_wo_cost_median = []
            ct_wo_cost_25p = []
            ct_wo_cost_75p = []
            #% Ohne Transaktionskosten
            ct_w_cost_sr = []
            ct_w_cost_alpha = []
            ct_w_cost_std = []
            ct_w_cost_mean = []
            ct_w_cost_gmean = []
            ct_w_cost_median = []
            ct_w_cost_25p = []
            ct_w_cost_75p = []
            #% Z-Score-Strategie
            #% Ohne Transaktionskosten
            zs_wo_cost_sr = []
            zs_wo_cost_alpha = []
            zs_wo_cost_std = []
            zs_wo_cost_mean = []
            zs_wo_cost_gmean = []
            zs_wo_cost_median = []
            zs_wo_cost_25p = []
            zs_wo_cost_75p = []
            #% Ohne Transaktionskosten
            zs_w_cost_sr = []
            zs_w_cost_alpha = []
            zs_w_cost_std = []
            zs_w_cost_mean = []
            zs_w_cost_gmean = []
            zs_w_cost_median = []
            zs_w_cost_25p = []
            zs_w_cost_75p = []

            #% Performance
            for bootstrap in bootstraps:

                #% Buy-and_Hold
                bootstrap_series = pd.Series(bootstrap)
                bootstrap_series.index = df_market[self.close_column].index
                bootstrap_series.name = "px_last"
                bootstrap_series = bootstrap_series.fillna(method="bfill")
                buy_and_hold_returns = mf.rate_of_return(
                    bootstrap_series, kind=self.return_kind
                )
                sr = mf.sharpe_ratio(buy_and_hold_returns)
                buy_and_hold_sr.append(sr)
                alpha = mf.calc_alpha(buy_and_hold_returns, risk_premium)
                buy_and_hold_alpha.append(alpha)
                std = np.std(buy_and_hold_returns)
                buy_and_hold_std.append(std)
                mean = np.mean(buy_and_hold_returns)
                buy_and_hold_mean.append(mean)
                g_mean = gmean((buy_and_hold_returns + 1).fillna(1)) - 1
                buy_and_hold_gmean.append(g_mean)
                median = np.percentile(buy_and_hold_returns, 50)
                buy_and_hold_median.append(median)
                percentile_25 = np.percentile(buy_and_hold_returns, 25)
                buy_and_hold_25p.append(percentile_25)
                percentile_75 = np.percentile(buy_and_hold_returns, 75)
                buy_and_hold_75p.append(percentile_75)

                #% Crossover gemäß Transition
                crossover_signal = mf.crossover_signal(
                    bootstrap_series,
                    avg_short=self.co_avg_short,
                    avg_long=self.co_avg_long,
                )
                co_wo_cost_returns, co_w_cost_returns = mf.returns_weights(
                    crossover_signal,
                    buy_and_hold_returns,
                    self.transitions,
                    kind=self.return_kind,
                    transaction_costs=self.transaction_costs,
                )
                #% Ohne Transaktionskosten
                sr = mf.sharpe_ratio(co_wo_cost_returns)
                co_wo_cost_sr.append(sr)
                alpha = mf.calc_alpha(co_wo_cost_returns, risk_premium)
                co_wo_cost_alpha.append(alpha)
                std = np.std(co_wo_cost_returns)
                co_wo_cost_std.append(std)
                mean = np.mean(co_wo_cost_returns)
                g_mean = gmean((co_wo_cost_returns + 1).fillna(1)) - 1
                co_wo_cost_gmean.append(g_mean)
                co_wo_cost_mean.append(mean)
                median = np.percentile(co_wo_cost_returns, 50)
                co_wo_cost_median.append(median)
                percentile_25 = np.percentile(co_wo_cost_returns, 25)
                co_wo_cost_25p.append(percentile_25)
                percentile_75 = np.percentile(co_wo_cost_returns, 75)
                co_wo_cost_75p.append(percentile_75)
                #% Mit Transaktionskosten
                sr = mf.sharpe_ratio(co_w_cost_returns)
                co_w_cost_sr.append(sr)
                alpha = mf.calc_alpha(co_w_cost_returns, risk_premium)
                co_w_cost_alpha.append(alpha)
                std = np.std(co_w_cost_returns)
                co_w_cost_std.append(std)
                mean = np.mean(co_w_cost_returns)
                g_mean = gmean((co_w_cost_returns + 1).fillna(1)) - 1
                co_w_cost_gmean.append(g_mean)
                co_w_cost_mean.append(mean)
                median = np.percentile(co_w_cost_returns, 50)
                co_w_cost_median.append(median)
                percentile_25 = np.percentile(co_w_cost_returns, 25)
                co_w_cost_25p.append(percentile_25)
                percentile_75 = np.percentile(co_w_cost_returns, 75)
                co_w_cost_75p.append(percentile_75)

                #% Countertrend gemäß Transition
                close_series = bootstrap_series.copy()
                rolling_std = (
                    close_series.rolling(self.std_window).std().fillna(method="bfill")
                )
                high_series = close_series + rolling_std
                low_series = close_series - rolling_std
                plunger_criteria_df = mf.clenow_counter_plunger_criteria(
                    high_series,
                    low_series,
                    close_series,
                    trend=self.trend,
                    mean_lower=self.ct_avg_short,
                    mean_higher=self.ct_avg_long,
                    window_reading=self.window_reading,
                    true_range=self.true_range,
                    plunge_trigger=self.plunge_trigger,
                )
                plunger_signal = mf.plunger_signal(
                    plunger_criteria_df, confirmation_period=self.confirmation_period
                )
                ct_wo_cost_returns, ct_w_cost_returns = mf.returns_weights(
                    plunger_signal,
                    buy_and_hold_returns,
                    self.transitions,
                    kind=self.return_kind,
                    transaction_costs=self.transaction_costs,
                )
                #% Ohne Transaktionskosten
                sr = mf.sharpe_ratio(ct_wo_cost_returns)
                ct_wo_cost_sr.append(sr)
                alpha = mf.calc_alpha(ct_wo_cost_returns, risk_premium)
                ct_wo_cost_alpha.append(alpha)
                std = np.std(ct_wo_cost_returns)
                ct_wo_cost_std.append(std)
                mean = np.mean(ct_wo_cost_returns)
                ct_wo_cost_mean.append(mean)
                g_mean = gmean((ct_wo_cost_returns + 1).fillna(1)) - 1
                ct_wo_cost_gmean.append(g_mean)
                median = np.percentile(ct_wo_cost_returns, 50)
                ct_wo_cost_median.append(median)
                percentile_25 = np.percentile(ct_wo_cost_returns, 25)
                ct_wo_cost_25p.append(percentile_25)
                percentile_75 = np.percentile(ct_wo_cost_returns, 75)
                ct_wo_cost_75p.append(percentile_75)
                #% Mit Transaktionskosten
                sr = mf.sharpe_ratio(ct_w_cost_returns)
                ct_w_cost_sr.append(sr)
                alpha = mf.calc_alpha(ct_w_cost_returns, risk_premium)
                ct_w_cost_alpha.append(alpha)
                std = np.std(ct_w_cost_returns)
                ct_w_cost_std.append(std)
                mean = np.mean(ct_w_cost_returns)
                g_mean = gmean((ct_w_cost_returns + 1).fillna(1)) - 1
                ct_w_cost_gmean.append(g_mean)
                ct_w_cost_mean.append(mean)
                median = np.percentile(ct_w_cost_returns, 50)
                ct_w_cost_median.append(median)
                percentile_25 = np.percentile(ct_w_cost_returns, 25)
                ct_w_cost_25p.append(percentile_25)
                percentile_75 = np.percentile(ct_w_cost_returns, 75)
                ct_w_cost_75p.append(percentile_75)

                # Z-Score-Strategie gemäß Transaktion
                zscore_signal = mf.z_score_signals(
                    bootstrap_series,
                    window=self.zscore_window,
                    period=self.zscore_period,
                )
                zs_wo_cost_returns, zs_w_cost_returns = mf.returns_weights(
                    zscore_signal,
                    buy_and_hold_returns,
                    self.transitions,
                    kind=self.return_kind,
                    transaction_costs=self.transaction_costs,
                )
                #% Ohne Transaktionskosten
                sr = mf.sharpe_ratio(zs_wo_cost_returns)
                zs_wo_cost_sr.append(sr)
                alpha = mf.calc_alpha(zs_wo_cost_returns, risk_premium)
                zs_wo_cost_alpha.append(alpha)
                std = np.std(zs_wo_cost_returns)
                zs_wo_cost_std.append(std)
                mean = np.mean(zs_wo_cost_returns)
                zs_wo_cost_mean.append(mean)
                g_mean = gmean((zs_wo_cost_returns + 1).fillna(1)) - 1
                zs_wo_cost_gmean.append(g_mean)
                median = np.percentile(zs_wo_cost_returns, 50)
                zs_wo_cost_median.append(median)
                percentile_25 = np.percentile(zs_wo_cost_returns, 25)
                zs_wo_cost_25p.append(percentile_25)
                percentile_75 = np.percentile(zs_wo_cost_returns, 75)
                zs_wo_cost_75p.append(percentile_75)
                #% Mit Transaktionskosten
                sr = mf.sharpe_ratio(zs_w_cost_returns)
                zs_w_cost_sr.append(sr)
                alpha = mf.calc_alpha(zs_w_cost_returns, risk_premium)
                zs_w_cost_alpha.append(alpha)
                std = np.std(zs_w_cost_returns)
                zs_w_cost_std.append(std)
                mean = np.mean(zs_w_cost_returns)
                g_mean = gmean((zs_w_cost_returns + 1).fillna(1)) - 1
                zs_w_cost_gmean.append(g_mean)
                zs_w_cost_mean.append(mean)
                median = np.percentile(zs_w_cost_returns, 50)
                zs_w_cost_median.append(median)
                percentile_25 = np.percentile(zs_w_cost_returns, 25)
                zs_w_cost_25p.append(percentile_25)
                percentile_75 = np.percentile(zs_w_cost_returns, 75)
                zs_w_cost_75p.append(percentile_75)

            bootstraps_results_dict = {
                "buy_and_hold_sr": buy_and_hold_sr,
                "buy_and_hold_alpha": buy_and_hold_alpha,
                "buy_and_hold_std": buy_and_hold_std,
                "buy_and_hold_mean": buy_and_hold_mean,
                "buy_and_hold_gmean": buy_and_hold_gmean,
                "buy_and_hold_median": buy_and_hold_median,
                "buy_and_hold_25p": buy_and_hold_25p,
                "buy_and_hold_75p": buy_and_hold_75p,
                "co_wo_cost_sr": co_wo_cost_sr,
                "co_wo_cost_alpha": co_wo_cost_alpha,
                "co_wo_cost_std": co_wo_cost_std,
                "co_wo_cost_mean": co_wo_cost_mean,
                "co_wo_cost_gmean": co_wo_cost_gmean,
                "co_wo_cost_median": co_wo_cost_median,
                "co_wo_cost_25p": co_wo_cost_25p,
                "co_wo_cost_75p": co_wo_cost_75p,
                "co_w_cost_sr": co_w_cost_sr,
                "co_w_cost_alpha": co_w_cost_alpha,
                "co_w_cost_std": co_w_cost_std,
                "co_w_cost_mean": co_w_cost_mean,
                "co_w_cost_gmean": co_w_cost_gmean,
                "co_w_cost_median": co_w_cost_median,
                "co_w_cost_25p": co_w_cost_25p,
                "co_w_cost_75p": co_w_cost_75p,
                "ct_wo_cost_sr": ct_wo_cost_sr,
                "ct_wo_cost_alpha": ct_wo_cost_alpha,
                "ct_wo_cost_std": ct_wo_cost_std,
                "ct_wo_cost_mean": ct_wo_cost_mean,
                "ct_wo_cost_gmean": ct_wo_cost_gmean,
                "ct_wo_cost_median": ct_wo_cost_median,
                "ct_wo_cost_25p": ct_wo_cost_25p,
                "ct_wo_cost_75p": ct_wo_cost_75p,
                "ct_w_cost_sr": ct_w_cost_sr,
                "ct_w_cost_alpha": ct_w_cost_alpha,
                "ct_w_cost_std": ct_w_cost_std,
                "ct_w_cost_mean": ct_w_cost_mean,
                "ct_w_cost_gmean": ct_w_cost_gmean,
                "ct_w_cost_median": ct_w_cost_median,
                "ct_w_cost_25p": ct_w_cost_25p,
                "ct_w_cost_75p": ct_w_cost_75p,
                "zs_wo_cost_sr": zs_wo_cost_sr,
                "zs_wo_cost_alpha": co_wo_cost_alpha,
                "zs_wo_cost_std": zs_wo_cost_std,
                "zs_wo_cost_mean": co_wo_cost_mean,
                "zs_wo_cost_gmean": zs_wo_cost_gmean,
                "zs_wo_cost_median": zs_wo_cost_median,
                "zs_wo_cost_25p": co_wo_cost_25p,
                "zs_wo_cost_75p": zs_wo_cost_75p,
                "zs_w_cost_sr": zs_w_cost_sr,
                "zs_w_cost_alpha": zs_w_cost_alpha,
                "zs_w_cost_std": zs_w_cost_std,
                "zs_w_cost_mean": zs_w_cost_mean,
                "zs_w_cost_gmean": zs_w_cost_gmean,
                "zs_w_cost_median": zs_w_cost_median,
                "zs_w_cost_25p": zs_w_cost_25p,
                "zs_w_cost_75p": zs_w_cost_75p,
            }

            bootstraps_results = pd.DataFrame(bootstraps_results_dict)
            #% Abspeichern der Ergebnisse
            os.chdir(self.wd_export)
            export_filename = (
                self.filename[:3] + "_bootstraps_" + self.export_kind + ".csv"
            )
            bootstraps_results.to_csv(export_filename, header=True)
