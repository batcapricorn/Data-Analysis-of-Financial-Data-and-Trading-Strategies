# Data Analysis of Financial Data and Trading Strategies
 
 This is the code utilized in the seminar **Data Decision Science** of the **University of Augsburg** (**Chair of Statistics**, Spring 2020). **Munich Re (Group)** accompanied the project.

# Background

It was investigated to what extent a cost-average effect can occur while following technical trading strategies. Looking at five different global markets, the investigations were based on common mean-reversion and constant-mix strategies. Due to data protection reasons, the respective results cannot be published on GitHub. 

# Python Scripts

 ## exploratory_data_analysis.py
 EDA of market data: first visualizations and data transformations (e.g. currency conversion).

 ## main_functions.py
 Numerous useful functions that were employed in this analysis. It can be seen as toolbox of this projects. It is imported by `signal_analysis_class.py`.

 ## signal_analysis_class.py
 Forms a class module employing the above-mentioned toolbox `main_functions.py`. This step helped to automate the process of analyzing different strategies, markets and trading rhythms.
  
 ## signal_analysis.py
 Executes the analysis implemented in `signal_analysis.py`.

 ## constant_mix_analysis.py
 Analyzes if a cost-average can occur while following different constant mix strategies.

 ## evaluation_constant_mix.py
 Summarizes the results retrieved by `constant_mix_analysis.py`.

 ### evaluation_signals.py
 Summarizes the results retrieved by `signal_analysis.py`.
 
 ## visualizations.py
 Different visualizations of our results used in the final presentation.
