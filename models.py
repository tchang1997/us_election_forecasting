import numpy as np
import pandas as pd
import pymc as pm

from aggregator import Aggregator, HistoricalAverage
from election_data import PollDataset
from plotting import plot_forecast

from typing import Optional

class ElectionForecaster(object):
    def __init__(
            self,
            time_window_days: Optional[int] = 1,
            election_date_col: Optional[str] = "electiondate",
            poll_date_col: Optional[str] = "polldate"
        ):
        self.time_window_days = time_window_days
        self.election_date_col = election_date_col 
        self.poll_date_col = poll_date_col

class PollOnlyModel(ElectionForecaster):
    def __init__(self, year: int, aggregator: Optional[Aggregator] = HistoricalAverage(), **kwargs):
        super().__init__(**kwargs)
        self.year = year
        self.aggregator = aggregator

    def get_windows_to_election(self, df: pd.DataFrame):
        tte = df[self.election_date_col] - df[self.poll_date_col]
        max_timedelta = tte.max()   
        bins = pd.timedelta_range(start='0D', end=max_timedelta + pd.Timedelta(days=self.time_window_days), freq=f'{self.time_window_days}D')
        windows_to_election = pd.cut(tte, bins=bins, labels=np.arange(len(bins) - 1)).astype(int)
        return windows_to_election, bins

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
            state_variance_prior_bias: Optional[float] = 1e-4,
            state_variance_prior_sigma: Optional[float] = 0.3,
            national_variance_prior_bias: Optional[float] = 1e-4,
            national_variance_prior_sigma: Optional[float] = 0.1,
            national_final_variance: Optional[float] = 1e-6,
        ): # default: one week out. max_windows_back should be deprecated

        print("Preprocessing data...")
        df = data.filter_polls(year=self.year)
        windows_to_election, time_coords = self.get_windows_to_election(df) # time-to-election index
        d_avg, r_avg = self.aggregator(data.filter_polls(), self.year)
        time_idx = windows_to_election[windows_to_election >= forecast_horizon] # note that n windows out = pi[n]
        loc_idx, loc = df.loc[windows_to_election >= forecast_horizon].set_index("location").index.factorize(sort=True)

        COORDS = {
            "location": loc,
            "time": time_coords.astype(str),
        }
        n_locations = len(df.location.unique())

        obs_n = df.loc[windows_to_election >= forecast_horizon, "samplesize"].astype(int) # sample sizes, for validation # TODO make all colnames keywords
        emp_dem_n = (df.loc[windows_to_election >= forecast_horizon, "cand1_pct"] * obs_n / 100).astype(int)
        emp_rep_n = (df.loc[windows_to_election >= forecast_horizon, "cand2_pct"] * obs_n / 100).astype(int)

        print("Initiating modeling...")
        with pm.Model(coords=COORDS) as model:
            # state level effect. Prior = HalfNormal(0, 1) + slight bias away from 0.
            dem_state_var = pm.HalfNormal('dem_time_var', sigma=state_variance_prior_sigma, initval=np.ones(n_locations), dims="location") + state_variance_prior_bias # different from Linzer (calls this beta_ij). But I do think a half-normal prior makes more sense
            rep_state_var = pm.HalfNormal('rep_time_var', sigma=state_variance_prior_sigma, initval=np.ones(n_locations), dims="location") + state_variance_prior_bias

            # National-level effect (delta)
            sigma_delta = pm.HalfNormal('national_effect', sigma=national_variance_prior_sigma) + national_variance_prior_bias # shared 
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
            _ = pm.Binomial('rep_obs', n=obs_n, p=rep_polling_param, observed=emp_rep_n)  

            trace = pm.sample(n_samples, tune=n_tune_steps, cores=n_cores, chains=n_chains, step=pm.NUTS(target_accept=target_accept, max_treedepth=max_treedepth)) # be able to configure this
            summary_df = pm.summary(trace)
        return model, trace, summary_df
        
    