from functools import reduce
import operator as op

import numpy as np
import pandas as pd
import pymc as pm

from aggregator import Aggregator, HistoricalAverage
from election_data import PollDataset, PollsterDataset, TwoPartyElectionResultDataset, TWO_PARTY_COLMAP
from plotting import plot_forecast
from utils import convert_col_to_date, VALID_LOCATIONS

from typing import Dict, List, Optional, Tuple, Union

class ElectionForecaster(object):
    def __init__(
            self,
            colmap,
            time_window_days: Optional[int] = 1,
            max_backcast: Optional[int] = 60, # use a max. of two months of polling data from before

        ):
        self.time_window_days = time_window_days
        self.max_backcast = max_backcast
        self.election_date_col = colmap["election_date"] 
        self.poll_date_col = colmap["poll_date"]
        self.dem_pct_col = colmap["dem_pct"]
        self.rep_pct_col = colmap["rep_pct"]
        self.location_col = colmap["location"]        
        self.sample_size_col = colmap["sample_size"]

class PollOnlyModel(ElectionForecaster):
    def __init__(self, year: int, colmap: Dict, aggregator: Optional[Aggregator.__class__] = HistoricalAverage, **kwargs):
        super().__init__(colmap, **kwargs)
        self.year = year
        self.aggregator = aggregator(TWO_PARTY_COLMAP)

    def get_windows_to_election(self, df: pd.DataFrame, forecast_horizon: int):
        if not pd.api.types.is_datetime64_any_dtype(df[self.election_date_col]):
            election_date = convert_col_to_date(df[self.election_date_col])
        if not pd.api.types.is_datetime64_any_dtype(df[self.poll_date_col]):
            poll_date = convert_col_to_date(df[self.poll_date_col])
        
        df["time_to_election"] = election_date - poll_date
        max_timedelta = min(df["time_to_election"].max(), pd.Timedelta(days=forecast_horizon + self.max_backcast))
        bins = pd.timedelta_range(start='0D', end=max_timedelta + pd.Timedelta(days=self.time_window_days), freq=f'{self.time_window_days}D')
        df["windows_to_election"] = pd.cut(df["time_to_election"], bins=bins, labels=np.arange(len(bins) - 1), include_lowest=True)
        #windows_to_election = binned.dropna().astype(int)
        return df
        #return windows_to_election, bins.astype(str), binned.isna()

    def get_coords_and_indices(self, orig_df, filtered_df):
        """
        Compute coordinates and indices for the PyMC model.

        This method generates the basic coordinates and indices for the poll-only model.
        Subclasses should override this method to add their own coordinates and indices
        specific to their model structure. Note that we need to treat time coordinates
        specially due to the structure of the data, so it's hard to directly generate them here.

        Returns:
            tuple: A tuple containing two dictionaries:
                - COORDS: Dictionary of coordinates for the PyMC model.
                - INDICES: Dictionary of indices for the PyMC model.
        """

        # other factors can be created by factorizing a column more directly, but we need to treat time and location specially. Time indices need to be pre-computed.
        # the time factors should correspond to a list of N-day windows 
        time = orig_df["windows_to_election"].dtype.categories
        time_idx = pd.Categorical(filtered_df["windows_to_election"], categories=time).codes
        
        #_, loc = pd.factorize(orig_df[self.location_col], sort=True)
        loc = np.array(sorted(VALID_LOCATIONS)) # for cases where there are states with no polls. Shouldn't happen unless you're looking at some weird window.
        loc_idx = pd.Categorical(filtered_df[self.location_col], categories=loc).codes
        COORDS = {
            "location": loc,
            "time": time,
        }
        INDICES = {
            "location": loc_idx,
            "time": time_idx
        }
        return COORDS, INDICES
    
    def create_variance_priors(
        self,
        locations: np.ndarray,
        prior_type: str,
        state_variance_prior_sigma: float,
        state_variance_prior_bias: float,
        national_variance_prior_sigma: float,
        national_variance_prior_bias: float
    ) -> Tuple[pm.distributions.continuous.PositiveContinuous, pm.distributions.continuous.PositiveContinuous, pm.distributions.continuous.PositiveContinuous]:
        # state level effect. Prior = HalfNormal(0, 1) + slight bias away from 0.
        state_variance_prior = np.ones(*locations.shape) * state_variance_prior_sigma
        state_variance_prior[np.where(locations == "US")[0].item()] = 1e-6 # State effect should be essentially zero for the "US" index -- we model everything with a shared covariance to allow state-level errors to correlate w/ national errors

        if prior_type == "half_normal":
            dem_state_var = pm.HalfNormal('dem_time_var', sigma=state_variance_prior_sigma, dims="location") + state_variance_prior_bias # different from Linzer (calls this beta_ij). But I do think a half-normal prior makes more sense
            rep_state_var = pm.HalfNormal('rep_time_var', sigma=state_variance_prior_sigma, dims="location") + state_variance_prior_bias
            sigma_delta = pm.HalfNormal('national_effect', sigma=national_variance_prior_sigma) + national_variance_prior_bias # shared. 
        elif prior_type == "half_t":
            dem_state_var = pm.HalfStudentT('dem_time_var', sigma=state_variance_prior_sigma, nu=3, dims="location") + state_variance_prior_bias # different from Linzer (calls this beta_ij). But I do think a half-normal prior makes more sense
            rep_state_var = pm.HalfStudentT('rep_time_var', sigma=state_variance_prior_sigma, nu=3, dims="location") + state_variance_prior_bias
            sigma_delta = pm.HalfStudentT('national_effect', sigma=national_variance_prior_sigma, nu=3) + national_variance_prior_bias # shared. 
        elif prior_type == "half_cauchy":
            # use the "sigma" named parameters as the beta parameter instead
            dem_state_var = pm.HalfCauchy('dem_time_var', beta=state_variance_prior_sigma, dims="location") + state_variance_prior_bias # different from Linzer (calls this beta_ij). But I do think a half-normal prior makes more sense
            rep_state_var = pm.HalfCauchy('rep_time_var', beta=state_variance_prior_sigma, dims="location") + state_variance_prior_bias
            sigma_delta = pm.HalfCauchy('national_effect', beta=national_variance_prior_sigma) + national_variance_prior_bias # shared. 
        return dem_state_var, rep_state_var, sigma_delta

    def create_effect_vars(self, data_dict, priors):
        dem_national_effects, rep_national_effects = self.create_national_effects(
            priors['national']['var'],
            priors['national']['final_var']
        )
        dem_state_effects, rep_state_effects = self.create_state_effects(
            data_dict["pred_prior_mean"]["dem"],
            data_dict["pred_prior_mean"]["rep"],
            priors['state']['n_locations'],
            priors['state']['prior_precision'],
            priors['state']['dem_var'],
            priors['state']['rep_var']
        )
        return [dem_national_effects, dem_state_effects], [rep_national_effects, rep_state_effects]
    
    def create_national_effects(self, sigma_delta: float, national_final_variance: float) -> Tuple[pm.GaussianRandomWalk, pm.GaussianRandomWalk]:
        dem_national_final = pm.Normal.dist(mu=0, sigma=national_final_variance)  # we want this to be constant, but sigma = 0 causes issues
        dem_national_effects = pm.GaussianRandomWalk('dem_national', sigma=sigma_delta, dims="time", init_dist=dem_national_final)
        rep_national_final = pm.Normal.dist(mu=0, sigma=national_final_variance)
        rep_national_effects = pm.GaussianRandomWalk('rep_national', sigma=sigma_delta, dims="time", init_dist=rep_national_final)
        return dem_national_effects, rep_national_effects
    
    def create_state_effects(
        self,
        d_avg: np.ndarray[float],
        r_avg: np.ndarray[float],
        n_locations: int,
        prior_precision: float,
        dem_state_var: pm.distributions.continuous.PositiveContinuous,
        rep_state_var: pm.distributions.continuous.PositiveContinuous,
    ) -> Tuple[pm.MvGaussianRandomWalk, pm.MvGaussianRandomWalk]:
        # State-level effect (beta) 
        dem_state_final = pm.MvNormal.dist(mu=pm.math.logit(d_avg), cov=np.eye(n_locations) / prior_precision)
        dem_state_effects = pm.MvGaussianRandomWalk('dem_time_effects', mu=0, cov=np.eye(n_locations) * dem_state_var, dims=("time", "location"), init_dist=dem_state_final) 
        rep_state_final = pm.MvNormal.dist(mu=pm.math.logit(r_avg), cov=np.eye(n_locations) / prior_precision) 
        rep_state_effects = pm.MvGaussianRandomWalk('rep_time_effects', mu=0, cov=np.eye(n_locations) * rep_state_var, dims=("time", "location"), init_dist=rep_state_final)
        return dem_state_effects, rep_state_effects
    
    def create_polling_parameters(
        self,
        indices: Dict[str, pd.Series],
        dem_params: List[Union[pm.MvGaussianRandomWalk, pm.GaussianRandomWalk]],
        rep_params: List[Union[pm.MvGaussianRandomWalk, pm.GaussianRandomWalk]],
    ) -> Tuple[pm.Deterministic, pm.Deterministic, pm.Deterministic, pm.Deterministic]:
        final_dem_vars = [var if var.ndim > 1 else var[:, np.newaxis] for var in dem_params] # original should have shape (time, location) or (time). So this ensures broadcasting works correctly.
        final_rep_vars = [var if var.ndim > 1 else var[:, np.newaxis] for var in rep_params]
        final_dem_expr = reduce(op.add, final_dem_vars)
        final_rep_expr = reduce(op.add, final_rep_vars)

        dem_pi = pm.Deterministic('dem_pi', pm.math.invlogit(final_dem_expr), dims=("time", "location"))
        rep_pi = pm.Deterministic('rep_pi', pm.math.invlogit(final_rep_expr), dims=("time", "location"))

        dem_polling_param = pm.Deterministic("dem_polling_param", dem_pi[indices["time"], indices["location"]])
        rep_polling_param = pm.Deterministic("rep_polling_param", rep_pi[indices["time"], indices["location"]])

        return dem_pi, rep_pi, dem_polling_param, rep_polling_param
    
    def create_likelihood(
        self,
        obs_n: pd.Series,
        dem_polling_param: pm.Deterministic,
        rep_polling_param: pm.Deterministic,
        emp_dem_n: pd.Series,
        emp_rep_n: pd.Series,
        polling_weights: Optional[pd.Series] = None
    ) -> Optional[pm.Potential]:
        if polling_weights is not None:
            dem_ll = pm.Binomial.dist(n=obs_n, p=dem_polling_param).logp(emp_dem_n)
            rep_ll = pm.Binomial.dist(n=obs_n, p=rep_polling_param).logp(emp_rep_n)
            weighted_log_likelihood = polling_weights * (dem_ll + rep_ll)  # each observation is weighted equally
            return pm.Potential('weighted_ll', weighted_log_likelihood)
        else:
            _ = pm.Binomial('dem_obs', n=obs_n, p=dem_polling_param, observed=emp_dem_n)
            _ = pm.Binomial('rep_obs', n=obs_n, p=rep_polling_param, observed=emp_rep_n)

    def apply_poll_exclusions(
        self,
        df: pd.DataFrame,
        forecast_horizon: int,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        df_time_subset = df.dropna(subset="windows_to_election")
        null_sample_size = pd.isnull(df_time_subset[self.sample_size_col])
        # TODO: filter out banned pollsters here

        poll_inclusion_criteria = (df_time_subset["windows_to_election"] >= forecast_horizon) & ~null_sample_size
        return df_time_subset.loc[poll_inclusion_criteria]
    
    def get_election_result_priors(self):
        election_results = TwoPartyElectionResultDataset()
        d_avg, r_avg = self.aggregator(election_results.data, self.year)  # make sure we only have valid locations + ensure sorting consistency
        return d_avg, r_avg

    def get_observed_poll_values(
        self,
        included_polls: pd.DataFrame,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        obs_n = included_polls[self.sample_size_col].astype(int)  # sample sizes, for validation
        emp_dem_n = (included_polls[self.dem_pct_col] * obs_n / 100).astype(int)
        emp_rep_n = (included_polls[self.rep_pct_col] * obs_n / 100).astype(int)
        start_date = included_polls[self.poll_date_col].min()
        end_date = included_polls[self.poll_date_col].max()
        print("Total:", len(obs_n), "polls found from", start_date, "to", end_date)
        return obs_n, emp_dem_n, emp_rep_n
    
    def preprocess_data(self, data: PollDataset, forecast_horizon: int):
        df = data.filter_polls(year=self.year)
        df_with_time = self.get_windows_to_election(df, forecast_horizon=forecast_horizon) # time-to-election index
        filtered_df = self.apply_poll_exclusions(df_with_time, forecast_horizon)
        coords, indices = self.get_coords_and_indices(df_with_time, filtered_df)

        # get observed poll values
        obs_n, emp_dem_n, emp_rep_n = self.get_observed_poll_values(filtered_df)
        polling_weights = None # TODO: compute one for each row of df_time_subset.loc[poll_inclusion_criteria], as a function of time and pollster rating 
        # compute final result prior mean
        d_avg, r_avg = self.get_election_result_priors()

        return {
            "params": {
                "coords": coords,
                "indices": indices,
            },
            "observed": {
                "n": obs_n, "dem": emp_dem_n, "rep": emp_rep_n, "weights": polling_weights
            },
            "pred_prior_mean": {
                "dem": d_avg, "rep": r_avg,
            }
        } 

    def fit(
            self,
            data: PollDataset,
            n_samples: Optional[int] = 2000,
            n_chains: Optional[int] = 2,
            n_cores: Optional[int] = 4,
            n_tune_steps: Optional[int] = 1000,
            target_accept: Optional[float] = 0.99,
            max_treedepth: Optional[int] = 13,
            forecast_horizon: Optional[int] = 7,
            prior_precision: Optional[float] = 5,
            prior_type: Optional[str] = "half_t",
            state_variance_prior_bias: Optional[float] = 1e-4,
            state_variance_prior_sigma: Optional[float] = 0.3,
            national_variance_prior_bias: Optional[float] = 1e-4,
            national_variance_prior_sigma: Optional[float] = 0.1,
            national_final_variance: Optional[float] = 1e-6,
        ):

        print("Preprocessing data...")
        data_dict = self.preprocess_data(data, forecast_horizon)
        n_locations = len(data_dict["params"]["coords"]["location"]) 
        with pm.Model(coords=data_dict["params"]["coords"]) as model:
            dem_state_var, rep_state_var, sigma_national = self.create_variance_priors(
                data_dict["params"]["coords"]["location"],
                prior_type,
                state_variance_prior_sigma, 
                state_variance_prior_bias, 
                national_variance_prior_sigma, 
                national_variance_prior_bias
            )

            # override this to incorporate fundamentals. TODO: how can programatically link this w/ create_variance_priors?
            priors = {
                'national': {
                    'var': sigma_national,
                    'final_var': national_final_variance
                },
                'state': {
                    'n_locations': n_locations,
                    'prior_precision': prior_precision,
                    'dem_var': dem_state_var,
                    'rep_var': rep_state_var
                }
            }
            dem_params, rep_params = self.create_effect_vars(data_dict, priors)
            _, _, dem_polling_param, rep_polling_param = self.create_polling_parameters(
                data_dict["params"]["indices"], 
                dem_params=dem_params, # we'd pass a larger list of effects to subclasses (e.g., to account for house effects)
                rep_params=rep_params,
            )
            self.create_likelihood(
                data_dict["observed"]["n"],
                dem_polling_param,
                rep_polling_param,
                data_dict["observed"]["dem"],
                data_dict["observed"]["rep"],
                data_dict["observed"]["weights"]
            )

            trace = pm.sample(
                n_samples,
                tune=n_tune_steps,
                cores=n_cores,
                chains=n_chains,
                step=pm.NUTS(target_accept=target_accept, max_treedepth=max_treedepth)
            )
            summary_df = pm.summary(trace)
        return model, trace, summary_df
        
class PollsAdjustedModel(ElectionForecaster): 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pollster_data = PollsterDataset()

    def get_coords_and_indices(self, orig_df, filtered_df):
        init_coords, init_indices = super().get_coords_and_indices(orig_df, filtered_df)
        # add pollster effects and variance prior
        final_coords, final_indices = {}, {}
        return final_coords, final_indices

    def apply_poll_exclusions(
        self,
        df: pd.DataFrame,
        forecast_horizon: int,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        df_time_subset = df.dropna(subset="windows_to_election")
        null_sample_size = pd.isnull(df_time_subset[self.sample_size_col])
        import pdb; pdb.set_trace() 
        # TODO: filter out banned pollsters here

        poll_inclusion_criteria = (df_time_subset["windows_to_election"] >= forecast_horizon) & ~null_sample_size
        return df_time_subset.loc[poll_inclusion_criteria] 
    




class PollsPlusAdjustedModel(ElectionForecaster):
    pass # fundamentals + house effects