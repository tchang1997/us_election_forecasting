# Constants for valid years and states
import numpy as np
import pandas as pd

VALID_ELECTION_YEARS = [2000, 2004, 2008, 2012, 2016, 2020]
VALID_US_STATES = {
    'AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI',
    'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME',
    'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM',
    'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
    'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY'
}

VALID_CONGRESSIONAL_DISTRICTS = {'M1', 'M2', 'N2'}
VALID_LOCATIONS = VALID_US_STATES.union(VALID_CONGRESSIONAL_DISTRICTS).union({"US"})

# Electoral votes by state for the 2024 election. TODO: Comprehensive hallucination check. Generated by Cursor. Spot-check seems to pass.
ELECTORAL_VOTES_2024 = {
    'AL': 9, 'AK': 3, 'AZ': 11, 'AR': 6, 'CA': 54, 'CO': 10, 'CT': 7, 'DE': 3,
    'DC': 3, 'FL': 30, 'GA': 16, 'HI': 4, 'ID': 4, 'IL': 19, 'IN': 11, 'IA': 6,
    'KS': 6, 'KY': 8, 'LA': 8, 'ME': 2, 'MD': 10, 'MA': 11, 'MI': 15, 'MN': 10,
    'MS': 6, 'MO': 10, 'MT': 4, 'NE': 4, 'NV': 6, 'NH': 4, 'NJ': 14, 'NM': 5,
    'NY': 28, 'NC': 16, 'ND': 3, 'OH': 17, 'OK': 7, 'OR': 8, 'PA': 19, 'RI': 4,
    'SC': 9, 'SD': 3, 'TN': 11, 'TX': 40, 'UT': 6, 'VT': 3, 'VA': 13, 'WA': 12,
    'WV': 4, 'WI': 10, 'WY': 3,
    'M1': 1, 'M2': 1, 'N2': 1, 
}
ELECTORAL_VOTES_2010 = {
    'AL': 9, 'AK': 3, 'AZ': 11, 'AR': 6, 'CA': 55, 'CO': 9, 'CT': 7, 'DE': 3,
    'DC': 3, 'FL': 29, 'GA': 16, 'HI': 4, 'ID': 4, 'IL': 20, 'IN': 11, 'IA': 6,
    'KS': 6, 'KY': 8, 'LA': 8, 'ME': 2, 'MD': 10, 'MA': 11, 'MI': 16, 'MN': 10,
    'MS': 6, 'MO': 10, 'MT': 3, 'NE': 4, 'NV': 6, 'NH': 4, 'NJ': 14, 'NM': 5,
    'NY': 29, 'NC': 15, 'ND': 3, 'OH': 18, 'OK': 7, 'OR': 7, 'PA': 20, 'RI': 4,
    'SC': 9, 'SD': 3, 'TN': 11, 'TX': 38, 'UT': 6, 'VT': 3, 'VA': 13, 'WA': 12,
    'WV': 5, 'WI': 10, 'WY': 3,
    'M1': 1, 'M2': 1, 'N2': 1, 
} # NE and ME must be adjusted manually to make sure things add up
TOTAL_EV = 538
EV_TO_WIN = int(TOTAL_EV / 2 + 1)

def is_valid_year(year, allow_none=True):
    return year in VALID_ELECTION_YEARS or allow_none * (year is None)

def is_valid_state(state, allow_none=True):
    return state in VALID_US_STATES or allow_none * (state is None)

def is_valid_location(state, allow_none=True):
    return state in VALID_LOCATIONS or allow_none * (state is None)

def preprocess_for_plotting(trace, polling_data, forecast_horizon, year, state, window_size=1):
    dem_pi = trace["posterior"]["dem_pi"]
    rep_pi = trace["posterior"]["rep_pi"]
    filtered_polling_data = polling_data.filter_polls(year=year, state=state)
    header = filtered_polling_data.iloc[0]
    election_result = polling_data.get_election_result(year=year, state=state)
    state_idx = np.where(trace["posterior"]["location"] == state)[0].item()
                                               
    tte = filtered_polling_data["electiondate"] - filtered_polling_data["polldate"]
    max_timedelta = tte.max()   
    bins = pd.timedelta_range(start='0D', end=max_timedelta + pd.Timedelta(days=window_size), freq=f'{window_size}D') # don't repeat self...sigh TODO: make less hacky
    windows_to_election = pd.cut(tte, bins=bins, labels=np.arange(len(bins) - 1)).astype(int)

    state_mask = (windows_to_election >= forecast_horizon) & (filtered_polling_data["location"] == state)
    polling_data_until_forecast = filtered_polling_data[state_mask]
    return dem_pi, rep_pi, forecast_horizon, header, state_idx, election_result, polling_data_until_forecast
