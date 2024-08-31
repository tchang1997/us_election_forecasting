import numpy as np
import pandas as pd
import pymc as pm

from aggregator import Aggregator, HistoricalAverage
from election_data import PollDataset, TwoPartyElectionResultDataset, TWO_PARTY_COLMAP
from plotting import plot_forecast

from typing import Dict, Optional, Tuple

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
            df[self.election_date_col] = pd.to_datetime(df[self.election_date_col], format='%m/%d/%y')
        if not pd.api.types.is_datetime64_any_dtype(df[self.poll_date_col]):
            df[self.poll_date_col] = pd.to_datetime(df[self.poll_date_col], format='%m/%d/%y')
        
        tte = df[self.election_date_col] - df[self.poll_date_col]
        max_timedelta = min(tte.max(), pd.Timedelta(days=forecast_horizon + self.max_backcast))
        bins = pd.timedelta_range(start='0D', end=max_timedelta + pd.Timedelta(days=self.time_window_days), freq=f'{self.time_window_days}D')
        binned = pd.cut(tte, bins=bins, labels=np.arange(len(bins) - 1), include_lowest=True)
        windows_to_election = binned.dropna().astype(int)
        return windows_to_election, bins.astype(str), binned.isna()
    
    def create_variance_priors(
        self,
        prior_type: str,
        state_variance_prior_sigma: float,
        state_variance_prior_bias: float,
        national_variance_prior_sigma: float,
        national_variance_prior_bias: float
    ) -> Tuple[pm.distributions.continuous.PositiveContinuous, pm.distributions.continuous.PositiveContinuous, pm.distributions.continuous.PositiveContinuous]:
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
        dem_state_effects: pm.MvGaussianRandomWalk,
        rep_state_effects: pm.MvGaussianRandomWalk,
        dem_national_effects: pm.GaussianRandomWalk,
        rep_national_effects: pm.GaussianRandomWalk,
        indices: Dict[str, pd.Series]
    ) -> Tuple[pm.Deterministic, pm.Deterministic, pm.Deterministic, pm.Deterministic]:
        dem_pi = pm.Deterministic('dem_pi', pm.math.invlogit(dem_state_effects + dem_national_effects[:, np.newaxis]), dims=("time", "location"))
        rep_pi = pm.Deterministic('rep_pi', pm.math.invlogit(rep_state_effects + rep_national_effects[:, np.newaxis]), dims=("time", "location"))

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
        data: PollDataset,
        forecast_horizon: int,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        # get polls for appropriate year
        df = data.filter_polls(year=self.year)

        # compute and filter by time to election 
        windows_to_election, time_coords, dropped = self.get_windows_to_election(
            df,
            forecast_horizon=forecast_horizon,
        ) # time-to-election index

        # get final cohort of polls
        df_time_subset = df.loc[~dropped]
        
        null_sample_size = pd.isnull(df_time_subset[self.sample_size_col])
        # TODO: filter out banned pollsters here
        poll_inclusion_criteria = (windows_to_election >= forecast_horizon) & ~null_sample_size
        time_idx = windows_to_election[poll_inclusion_criteria] # note that n windows out = pi[n]
        return df_time_subset, time_idx, time_coords, poll_inclusion_criteria
    
    def get_observed_poll_values(
        self,
        included_polls: pd.DataFrame,
        poll_inclusion_criteria: pd.Series
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        obs_n = included_polls.loc[poll_inclusion_criteria, self.sample_size_col].astype(int)  # sample sizes, for validation
        emp_dem_n = (included_polls.loc[poll_inclusion_criteria, self.dem_pct_col] * obs_n / 100).astype(int)
        emp_rep_n = (included_polls.loc[poll_inclusion_criteria, self.rep_pct_col] * obs_n / 100).astype(int)
        return obs_n, emp_dem_n, emp_rep_n

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
        ): # default: one week out. max_windows_back should be deprecated

        print("Preprocessing data...")
        # applies exclusion criteria based on forecast horizon, how far back to consider polls, and others (e.g. null sample size, banned pollsters)
        included_polls, time_idx, time_coords, poll_inclusion_criteria = self.apply_poll_exclusions(data, forecast_horizon)

        polling_weights = None # TODO: compute one for each row of df_time_subset.loc[poll_inclusion_criteria], as a function of time and pollster rating 
        
        # get observed poll values
        obs_n, emp_dem_n, emp_rep_n = self.get_observed_poll_values(included_polls, poll_inclusion_criteria)
        print("Total:", len(obs_n), "polls found")
        print("Initiating modeling...")

        # compute final result prior mean
        election_results = TwoPartyElectionResultDataset()
        d_avg, r_avg = self.aggregator(election_results.data, self.year) # make sure we only have valid locations + ensure sorting consistency

        # compute time and state coordinates. May differ across subclasses
        loc_idx, loc = included_polls.loc[poll_inclusion_criteria].set_index(self.location_col).index.factorize(sort=True) # other factors can be created by factorizing a column
        COORDS = {
            "location": loc,
            "time": time_coords,
        }
        INDICES = {
            "location": loc_idx,
            "time": time_idx
        }
        n_locations = len(COORDS["location"])

        with pm.Model(coords=COORDS) as model:
            # state level effect. Prior = HalfNormal(0, 1) + slight bias away from 0.
            state_variance_prior = np.ones(n_locations) * state_variance_prior_sigma
            state_variance_prior[np.where(loc == "US")[0].item()] = 1e-6

            dem_state_var, rep_state_var, sigma_delta = self.create_variance_priors(
                prior_type,
                state_variance_prior, 
                state_variance_prior_bias, 
                national_variance_prior_sigma, 
                national_variance_prior_bias
            )

            # National-level effect (delta)            
            dem_national_effects, rep_national_effects = self.create_national_effects(sigma_delta, national_final_variance)
            dem_state_effects, rep_state_effects = self.create_state_effects(d_avg, r_avg, n_locations, prior_precision, dem_state_var, rep_state_var)
            # here, we can add additional effects in subclasses

            # This function would be overridden in subclasses
            _, _, dem_polling_param, rep_polling_param = self.create_polling_parameters(
                dem_state_effects, 
                rep_state_effects, 
                dem_national_effects, 
                rep_national_effects, 
                INDICES, 
            )
            self.create_likelihood(obs_n, dem_polling_param, rep_polling_param, emp_dem_n, emp_rep_n, polling_weights)

            trace = pm.sample(n_samples, tune=n_tune_steps, cores=n_cores, chains=n_chains, step=pm.NUTS(target_accept=target_accept, max_treedepth=max_treedepth)) # be able to configure this
            summary_df = pm.summary(trace)
        return model, trace, summary_df
        
class PollsAdjustedModel(ElectionForecaster): 
    pass # house effects, but no fundamentals

class PollsPlusAdjustedModel(ElectionForecaster):
    pass # fundamentals + house effects