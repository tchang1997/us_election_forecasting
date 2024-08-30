import pandas as pd

from utils import is_valid_location, is_valid_year

class PollDataset(object):
    def __init__(self, path="./data/Rawpolls_061224.xlsx", sheet_name="rawpolls"):
        self.data = pd.read_excel(path, sheet_name=sheet_name)

    def filter_polls(self, type="Pres-G", year=None, state=None):
        if not is_valid_location(state) or not is_valid_year(year):
            raise ValueError(f"Invalid year or state: year={year}, state={state}")
        pres_polls = self.data[self.data["type_simple"] == type]
        if year is not None and state is not None:
            pres_polls = pres_polls[(pres_polls['year'] == year) & (pres_polls['location'] == state)]
        elif year is not None: # but state is None
            pres_polls = pres_polls[pres_polls['year'] == year]
        elif state is not None: # but year is None
            pres_polls = pres_polls[pres_polls['location'] == state] 
        return pres_polls
    
    def get_election_result(self, year, state, type="Pres-G"):
        if not is_valid_location(state) or not is_valid_year(year):
            raise ValueError(f"Invalid year or state: year={year}, state={state}") 
        row_mask = (self.data["type_simple"] == type) & (self.data["year"] == year) & (self.data["location"] == "state")
        race_polls = self.data.loc[row_mask, ["cand1_actual", "cand2_actual"]]
        results = race_polls.drop_duplicates()
        return results

class PollsterDataset(object):
    def __init__(self, path="./data/Pollster_Stats_Full_2024.xlsx", sheet_name="pollster-stats-full-june-2024"):
        self.data = pd.read_excel(path, sheet_name=sheet_name)

    def get_allowed_pollsters(self):
        return self.data[self.data["Banned by 538"] == "no"]