import pandas as pd

from typing import Optional, Tuple

class Aggregator:
    def __init__(self, dem_col: Optional[str] = "cand1_pct", rep_col: Optional[str] = "cand2_pct"):
        self.dem_col = dem_col
        self.rep_col = rep_col

class NaiveAverage(Aggregator): # essentially useless.
    def __call__(self, df: pd.DataFrame, wte: pd.Series, forecast_horizon: int, max_windows_back: int) -> Tuple[float, float]:
        history = df[(wte >= forecast_horizon)]
        return history[self.dem_col].mean() / 100, history[self.rep_col].mean()
    

class HistoricalAverage(Aggregator): # average of past election results, historical PVI

    def __call__(self, df: pd.DataFrame, max_year: int) -> Tuple[float, float]:
        state_results = df.groupby(["year", "location"])[["cand1_actual", "cand2_actual"]].mean().reset_index() # TODO: smarter aggregation; e.g. look for trends, discount outliers
        history = state_results[state_results["year"] < max_year]
        d_avg = history.groupby("location")["cand1_actual"].mean() / 100 # TODO: make this prior customizable
        r_avg = history.groupby("location")["cand2_actual"].mean() / 100
        return d_avg, r_avg
    
class CookPVIAverage(Aggregator):
    def __call__(self, *args):
        pass # TODO: implement this as a new prior -- they aggregate past election results somehow...