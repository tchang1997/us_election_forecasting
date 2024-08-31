import numpy as np
import pandas as pd
import pymc as pm

from aggregator import Aggregator, HistoricalAverage
from election_data import PollDataset, TwoPartyElectionResultDataset, TWO_PARTY_COLMAP
from plotting import plot_forecast

from typing import Dict, Optional

class ElectionForecaster(object):
    def __init__(
            self,
            colmap,
            time_window_days: Optional[int] = 1,
        ):
        self.time_window_days = time_window_days
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

    def get_windows_to_election(self, df: pd.DataFrame, forecast_horizon: int, max_backcast: int):
        if not pd.api.types.is_datetime64_any_dtype(df[self.election_date_col]):
            df[self.election_date_col] = pd.to_datetime(df[self.election_date_col], format='%m/%d/%y')
        if not pd.api.types.is_datetime64_any_dtype(df[self.poll_date_col]):
            df[self.poll_date_col] = pd.to_datetime(df[self.poll_date_col], format='%m/%d/%y')
        
        tte = df[self.election_date_col] - df[self.poll_date_col]
        max_timedelta = min(tte.max(), pd.Timedelta(days=forecast_horizon + max_backcast))
        bins = pd.timedelta_range(start='0D', end=max_timedelta + pd.Timedelta(days=self.time_window_days), freq=f'{self.time_window_days}D')
        binned = pd.cut(tte, bins=bins, labels=np.arange(len(bins) - 1), include_lowest=True)
        windows_to_election = binned.dropna().astype(int)
        return windows_to_election, bins, binned.isna()
    
    def create_variance_priors(self, prior_type, state_variance_prior_sigma, state_variance_prior_bias, national_variance_prior_sigma, national_variance_prior_bias):
        if prior_type == "half_normal":
            dem_state_var = pm.HalfNormal('dem_time_var', sigma=state_variance_prior_sigma, dims="location") + state_variance_prior_bias # different from Linzer (calls this beta_ij). But I do think a half-normal prior makes more sense
            rep_state_var = pm.HalfNormal('rep_time_var', sigma=state_variance_prior_sigma, dims="location") + state_variance_prior_bias
            sigma_delta = pm.HalfNormal('national_effect', sigma=national_variance_prior_sigma) + national_variance_prior_bias # shared. 
        elif prior_type == "half_t":
            dem_state_var = pm.HalfStudentT('dem_time_var', sigma=state_variance_prior_sigma, nu=3, dims="location") + state_variance_prior_bias # different from Linzer (calls this beta_ij). But I do think a half-normal prior makes more sense
            rep_state_var = pm.HalfStudentT('rep_time_var', sigma=state_variance_prior_sigma, nu=3, dims="location") + state_variance_prior_bias
            sigma_delta = pm.HalfStudentT('national_effect', sigma=national_variance_prior_sigma, nu=3) + national_variance_prior_bias # shared. 
        return dem_state_var, rep_state_var, sigma_delta

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
            max_backcast: Optional[int] = 60, # use a max. of two months of polling data from before
        ): # default: one week out. max_windows_back should be deprecated

        print("Preprocessing data...")
        df = data.filter_polls(year=self.year)
        windows_to_election, time_coords, dropped = self.get_windows_to_election(
            df,
            forecast_horizon=forecast_horizon,
            max_backcast=max_backcast
        ) # time-to-election index
        election_results = TwoPartyElectionResultDataset()
        d_avg, r_avg = self.aggregator(election_results.data, self.year) # make sure we only have valid locations + ensure sorting consistency

        df_time_subset = df.loc[~dropped]
        null_sample_size = pd.isnull(df_time_subset[self.sample_size_col])
        poll_inclusion_criteria = (windows_to_election >= forecast_horizon) & ~null_sample_size

        time_idx = windows_to_election[poll_inclusion_criteria] # note that n windows out = pi[n]
        loc_idx, loc = df_time_subset.loc[poll_inclusion_criteria].set_index(self.location_col).index.factorize(sort=True)

        COORDS = {
            "location": loc,
            "time": time_coords.astype(str),
        }
        n_locations = len(loc)

        obs_n = df_time_subset.loc[poll_inclusion_criteria, self.sample_size_col].astype(int) # sample sizes, for validation # TODO make all colnames keywords
        emp_dem_n = (df_time_subset.loc[poll_inclusion_criteria, self.dem_pct_col] * obs_n / 100).astype(int)
        emp_rep_n = (df_time_subset.loc[poll_inclusion_criteria, self.rep_pct_col] * obs_n / 100).astype(int)
        print("Total:", len(obs_n), "polls found")

        print("Initiating modeling...")
        with pm.Model(coords=COORDS) as model:
            # state level effect. Prior = HalfNormal(0, 1) + slight bias away from 0.
            state_variance_prior = np.ones(n_locations) * state_variance_prior_sigma
            state_variance_prior[np.where(loc == "US")[0].item()] = 1e-6

            dem_state_var, rep_state_var, sigma_delta = self.create_variance_priors(
                prior_type,
                state_variance_prior_sigma, 
                state_variance_prior_bias, 
                national_variance_prior_sigma, 
                national_variance_prior_bias
            )

            # National-level effect (delta)            
            dem_national_final = pm.Normal.dist(mu=0, sigma=national_final_variance) # we want this to be constant, but sigma = 0 causes issues
            dem_national_effects = pm.GaussianRandomWalk('dem_national', sigma=sigma_delta, dims="time", init_dist=dem_national_final)
            rep_national_final = pm.Normal.dist(mu=0, sigma=national_final_variance) 
            rep_national_effects = pm.GaussianRandomWalk('rep_national', sigma=sigma_delta, dims="time", init_dist=rep_national_final)

            # State-level effect (beta) 
            dem_state_final = pm.MvNormal.dist(mu=pm.math.logit(d_avg), cov=np.eye(n_locations) / prior_precision)
            dem_state_effects = pm.MvGaussianRandomWalk('dem_time_effects', mu=0, cov=np.eye(n_locations) * dem_state_var, dims=("time", "location"), init_dist=dem_state_final) 
            rep_state_final = pm.MvNormal.dist(mu=pm.math.logit(r_avg), cov=np.eye(n_locations) / prior_precision) 
            rep_state_effects = pm.MvGaussianRandomWalk('rep_time_effects', mu=0, cov=np.eye(n_locations) * rep_state_var, dims=("time", "location"), init_dist=rep_state_final)

            dem_pi = pm.Deterministic('dem_pi', pm.math.invlogit(dem_state_effects + dem_national_effects[:, np.newaxis]), dims=("time", "location")) # !! reversed in time! Note that the calculation of the Bernoulli **parameter** is deterinistic
            rep_pi = pm.Deterministic('rep_pi', pm.math.invlogit(rep_state_effects + rep_national_effects[:, np.newaxis]), dims=("time", "location"))

            dem_polling_param = pm.Deterministic("dem_polling_param", dem_pi[time_idx, loc_idx]) 
            rep_polling_param = pm.Deterministic("rep_polling_param", rep_pi[time_idx, loc_idx])

            # Likelihood (using observed proportions)
            _ = pm.Binomial('dem_obs', n=obs_n, p=dem_polling_param, observed=emp_dem_n)
            _ = pm.Binomial('rep_obs', n=obs_n, p=rep_polling_param, observed=emp_rep_n) # TODO: replace with pm.Potential to weight polls 
            trace = pm.sample(n_samples, tune=n_tune_steps, cores=n_cores, chains=n_chains, step=pm.NUTS(target_accept=target_accept, max_treedepth=max_treedepth)) # be able to configure this
            summary_df = pm.summary(trace)
        return model, trace, summary_df
        
    