# Data Decision Science

"""
Timimng-Analyse hinsichtlich der Umsetzung von verschiedenen technischen 
Anllagestrategien anhand des Moduls signal_analysis_class.py
"""

# Eigene Module
import b_main_functions as mf
import c_signal_analysis_class as sac

# Allgemeine Module
import pandas as pd
import os
import numpy as np
import glob
import time
from multiprocessing import Process

# Variablen
#% Allgemein
wd_main = os.getcwd()
wd_data = wd_main + "\\Data"
wd_other_data = wd_main + "\\Other_Data"
wd_export = wd_main + "\\Export"

#% Methodische Variablen
transitions = {
    "full_transition": {"days": [1], "portions": [1]},
    "linear_2d": {"days": [1, 2], "portions": [0.5, 1]},
    "linear_3d": {"days": [1, 2, 3], "portions": [0.33, 0.67, 1]},
    "progressive_2d": {"days": [1, 2], "portions": [0.6, 1]},
    "progressive_3d": {"days": [1, 2, 3], "portions": [0.5, 0.8, 1]},
}

bootstraps_replicates = 5

export_benchmark = True
export_bootstraps = False

# ----------- Ausführung ----------#
# Auslesen der Dateinamen
os.chdir(wd_data)
filenames = glob.glob("*.csv")
os.chdir(wd_main)

#% Benchmark Process
def benchmark_process(filename):
    """Funktion, die die jeweiligen Benchmark-Daten ohne Berücksichtigung von Bootstraps errechnet und ausgibt."""
    #% Set class
    signal_class = sac.signals_analysis(
        filename, wd_main, wd_data, wd_other_data, wd_export
    )

    #% Export der Benchmark-Daten
    for transition in transitions.keys():
        signal_class.export_kind = (
            transition  # Transition fließt in den Exportnamen ein
        )
        signal_class.transitions = transitions[transition]  # Transition
        signal_class.main_benchmark()


#% Ausgeben der Benchmark Ergebnisse
if (__name__ == "__main__") & (export_benchmark == True):
    start = time.time()
    for filename in filenames:
        p = Process(target=benchmark_process, args=(filename,))
        p.run()
    print(time.time() - start)

#% Bootstrap Process
def bootstrap_process(filename):
    """Funktion, die die jeweiligen Bootstrap-Daten errechnet und ausgibt."""
    #% Set class
    signal_class = sac.signals_analysis(
        filename, wd_main, wd_data, wd_other_data, wd_export
    )

    #% Export der Benchmark-Daten
    for transition in transitions.keys():
        signal_class.export_kind = (
            transition  # Transition fließt in den Exportnamen ein
        )
        signal_class.transitions = transitions[transition]  # value of transition-key
        signal_class.bootstraps_replicates = bootstraps_replicates
        signal_class.main_bootstraps()


#% Ausgeben der Bootstrap Ergebnisse
if (__name__ == "__main__") & (export_bootstraps == True):
    start = time.time()
    # for filename in filenames:
    filename = filenames[0]
    p = Process(target=bootstrap_process, args=(filename,))
    p.run()
    print(time.time() - start)
