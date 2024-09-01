import os

import numpy as np
import pandas as pd

from utils import is_valid_location, is_valid_year, TWO_PARTY_CANDIDATES, STATE_ABBREVIATIONS


TWO_PARTY_COLMAP = {
    "dem_actual": "DEM_actual",
    "rep_actual": "REP_actual",
    "location": "state_po",
    "year": "year"
}
class PollDataset(object):
    def __init__(
        self,
        path: str = "./data/Rawpolls_061224.xlsx",
        sheet_name: str = "rawpolls",
    ):
        self.path = path
        _, ext = os.path.splitext(path)
        if ext == ".xlsx":
            self.data = pd.read_excel(path, sheet_name=sheet_name)
        elif ext == ".csv":
            self.data = pd.read_csv(path, low_memory=False)
            
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    

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

    
class FiveThirtyEightPollDataset(PollDataset):
    def __init__(
        self,
        path: str = "./data/president_general_polls_2016.csv",
    ):
        super().__init__(path)
        self.data = self.data[self.data["type"] == "polls-only"] # 2016 538 data has polls in triplicate for each version of adjusted values
        
        # convert state names to state codes
        self.data["state_cd"] = self.data["state"].map(STATE_ABBREVIATIONS).fillna("US")

        self.election_results = TwoPartyElectionResultDataset()
        results_only = self.election_results.data.loc[:, ["year", "state_po", "DEM_actual", "REP_actual"]]
        # join using state_po <-> state_cd and year <-> cycle as a key
        self.data = pd.merge(
            self.data,
            results_only,
            left_on=['cycle', 'state_cd'],
            right_on=['year', 'state_po'],
            how='left'
        )

    def filter_polls(self, year=None, state=None):
        if not is_valid_location(state) or not is_valid_year(year):
            raise ValueError(f"Invalid year or state: year={year}, state={state}")
        if year is not None and state is not None:
            pres_polls = self.data[(self.data['cycle'] == year) & (self.data['state_cd'] == state)]
        elif year is not None: # but state is None
            pres_polls = self.data[self.data['cycle'] == year]
        elif state is not None: # but year is None
            pres_polls = self.data[self.data['state_cd'] == state] 
        return pres_polls
    
        
    def get_election_result(self, year, state):
        return self.election_results.get_election_result(year, state)
    
class FlatPollDataset(PollDataset): # converts a "one candidate per row" dataset into the format of PollDataset
    def __init__(self, path="./data/president_polls_historical.csv", **result_paths):
        super().__init__(path)
        self.path = path
        df = self.data[self.data["answer"].isin(TWO_PARTY_CANDIDATES)].dropna(how="all", axis=1)
        df_pivot = df.pivot(
            index=[col for col in df.columns if col not in ['candidate_name', 'pct', 'candidate_id', 'party', 'answer']],
            columns='party',  # Pivot on the party column to create new columns for each candidate
            values=['candidate_name', 'pct', 'candidate_id', 'answer']
        ).reset_index()
        df_pivot.columns = [f"{col1}_{col2}" if col2 else col1 for col1, col2 in df_pivot.columns]
        self.data = df_pivot.dropna(subset=["candidate_name_DEM", "candidate_name_REP"]).reset_index(drop=True)
        self.data["state_cd"] = self.data["state"].map(STATE_ABBREVIATIONS).fillna("US")

        # merge election results
        self.election_results = TwoPartyElectionResultDataset(**result_paths)
        results_only = self.election_results.data.loc[:, ["year", "state_po", "DEM_actual", "REP_actual"]]
        # join using state_po <-> state_cd and year <-> cycle as a key
        self.data = pd.merge(
            self.data,
            results_only,
            left_on=['cycle', 'state_cd'],
            right_on=['year', 'state_po'],
            how='left'
        )

    # TODO: refactor these two into some mixin, perhaps?
    def filter_polls(self, year=None, state=None):
        if not is_valid_location(state) or not is_valid_year(year):
            raise ValueError(f"Invalid year or state: year={year}, state={state}")
        pres_polls = self.data
        if year is not None and state is not None:
            pres_polls = self.data[(self.data['cycle'] == year) & (self.data['state_cd'] == state)]
        elif year is not None: # but state is None
            pres_polls = self.data[self.data['cycle'] == year]
        elif state is not None: # but year is None
            pres_polls = self.data[self.data['state_cd'] == state] 
        return pres_polls
    
    def get_election_result(self, year, state):
        return self.election_results.get_election_result(year, state)


class PollsterDataset(object):
    def __init__(self, path="./data/Pollster_Stats_Full_2024.xlsx", sheet_name="pollster-stats-full-june-2024"):
        self.data = pd.read_excel(path, sheet_name=sheet_name)

    def get_allowed_pollsters(self):
        return self.data[self.data["Banned by 538"] == "no"]
    
class TwoPartyElectionResultDataset(object):
    def __init__(self, path="./data/1976-2020-president.csv", congressional_results="./data/Rawpolls_061224.xlsx"): # we'll peek at this for the CG results
        self.path = path
        self.raw_data = pd.read_csv(path)
        df = self.raw_data.loc[
            (self.raw_data["party_simplified"].isin(["DEMOCRAT", "REPUBLICAN"])) &
            ((~(self.raw_data["writein"] == True)) | # some weird stuff happens in AZ and MD in the data
            ((self.raw_data["state"] == "DISTRICT OF COLUMBIA") & (self.raw_data["year"] == 2020))) # due to what appears to be a dataset coding error, all DC 2020 candidates are recorded as write-ins
        ].drop(columns="party_detailed").copy() # avoid SettingByCopyWarning
        df_pivot = df.pivot(
            index=[col for col in df.columns if col not in ['candidate', 'candidatevotes', 'party_simplified']],
            columns='party_simplified',  # Pivot on the party column to create new columns for each candidate
            values=['candidate', 'candidatevotes']
        ).reset_index()
        df_pivot.columns = [f"{col1}_{col2}" if col2 else col1 for col1, col2 in df_pivot.columns]
        df_final = df_pivot.dropna(how="all", axis=1)
        # Tabulate national votes
        national_results = df_final.groupby("year").agg({
            'totalvotes': 'sum',
            'candidatevotes_DEMOCRAT': 'sum',
            'candidatevotes_REPUBLICAN': 'sum',
            'office': 'last',
            'candidate_DEMOCRAT': 'last',
            'candidate_REPUBLICAN': 'last',
            'state': lambda _: 'NATIONAL',
            'state_po': lambda _: 'US',
            'year': 'last',
            **{col: lambda _: np.nan for col in df_final.columns if col not in [
                'totalvotes', 'candidatevotes_DEMOCRAT', 'candidatevotes_REPUBLICAN',
                'office', 'candidate_DEMOCRAT', 'candidate_REPUBLICAN',
                'state', 'state_po', 'year']}
        }).reset_index(drop=True)
        df_with_national = pd.concat([df_final, national_results])

        # Now compute percentages 
        df_with_national["DEM_actual"] = df_with_national["candidatevotes_DEMOCRAT"] / df_with_national["totalvotes"] * 100
        df_with_national["REP_actual"] = df_with_national["candidatevotes_REPUBLICAN"] / df_with_national["totalvotes"] * 100

        # concatenate NE and ME district results
        cg_df = pd.read_excel(congressional_results)
        cg_df = cg_df.loc[
            (cg_df["type_simple"] == "Pres-G") & 
            (cg_df["location"].str.contains(r'\d$')), 
            ["cand1_actual", "cand2_actual", "year", "location"]
        ].drop_duplicates()

        cg_df = cg_df.rename(columns={
            "cand1_actual": "DEM_actual",
            "cand2_actual": "REP_actual",
            "location": "state_po",
        })
        final_results = pd.concat([df_with_national, cg_df], ignore_index=True)

        # Fill NaN values for specified columns
        columns_to_fill = ['office', 'writein', 'candidate_DEMOCRAT', 'candidate_REPUBLICAN']
        final_results[columns_to_fill] = final_results.groupby('year')[columns_to_fill].ffill()
        final_results[columns_to_fill] = final_results.groupby('year')[columns_to_fill].bfill()

        self.data = final_results

    def get_election_result(self, year, state):
       return self.data.loc[(self.data["state_po"] == state) & (self.data["year"] == year), ["DEM_actual", "REP_actual"]]