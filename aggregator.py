import pandas as pd

from utils import VALID_LOCATIONS

from typing import Optional, Tuple

class Aggregator:
    def __init__(self, colmap):
        self.dem_actual_col = colmap["dem_actual"]
        self.rep_actual_col = colmap["rep_actual"]
        self.location_col = colmap["location"]
        self.year_col = colmap["year"]

class NaiveAverage(Aggregator): # essentially useless.
    def __call__(self, df: pd.DataFrame, tte: pd.Series, forecast_horizon: int, max_windows_back: int) -> Tuple[float, float]:
        """
        Calculate a naive average of polling data for a given forecast horizon.

        Args:
            df (pd.DataFrame): DataFrame containing polling data.
            tte (pd.Series): Series representing the 'time to election' for each poll.
            forecast_horizon (int): Number of days before the election to forecast.
            max_windows_back (int): Maximum number of time windows to look back.

        Returns:
            Tuple[float, float]: Two floats representing the average Democratic and Republican
            vote shares, respectively. Democratic vote share is divided by 100 to be between 0 and 1.
        """
        history = df[(tte >= forecast_horizon)]
        return history[self.dem_col].mean() / 100, history[self.rep_col].mean() / 100
    

class HistoricalAverage(Aggregator): # average of past election results, historical PVI

    def __call__(self, df: pd.DataFrame, max_year: int, years_back: Optional[int] = 8) -> Tuple[float, float]:
        """
        Calculate the historical average of election results for each location.

        There's several limitations to this kind of aggregation; i.e., anchoring to the raw vote percentages. For example, it could overindex to
        popular candidates (e.g., 2008 Obama), or deflate the priors in years with unusually strong third-parties (e.g., 2016). A future version
        might apply some "regression towards AR mean" type smoothing to ensure that outlier-like results don't influence the prior too severely.

        Args:
            df (pd.DataFrame): DataFrame containing historical election data.
            max_year (int): The most recent year to consider (exclusive).
            years_back (Optional[int]): Number of years to look back. Defaults to 8, making this a vanilla Cook Partisan Index-like weighting (two cycles back)

        Returns:
            Tuple[float, float]: Two Series containing the average Democratic and Republican
            vote shares for each location, respectively. Values are between 0 and 1.
        """
        state_results = df.groupby([self.year_col, self.location_col])[[self.dem_actual_col, self.rep_actual_col]].mean().reset_index() # TODO: smarter aggregation; e.g. look for trends, discount outliers
        history = state_results[
            (state_results[self.year_col] < max_year) & 
            (state_results[self.year_col] >= max_year - years_back) & 
            state_results[self.location_col].isin(VALID_LOCATIONS)
        ]
        d_avg = history.groupby(self.location_col)[self.dem_actual_col].mean() / 100 # TODO: make this prior customizable
        r_avg = history.groupby(self.location_col)[self.rep_actual_col].mean() / 100
        return d_avg.astype(float), r_avg.astype(float)
    