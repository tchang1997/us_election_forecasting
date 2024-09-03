import argparse
import operator as op
import os

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from ruamel.yaml import YAML

from election_data import PollDataset, FlatPollDataset, FiveThirtyEightPollDataset
import models
from plotting import plot_forecast
from reporting import calculate_ev_forecast, calculate_forecast_metrics, print_forecast_table
from utils import preprocess_for_plotting, VALID_LOCATIONS

def evaluate_forecast(
        trace,
        colmap,
        df: pd.DataFrame,
        results_dir: str,
    ):
    locations = trace.posterior.coords["location"].values
    pred_dem = trace.posterior.dem_pi[:, :, 0, :] # dims: chain, sample_size, time (reversed), location 
    pred_rep = trace.posterior.rep_pi[:, :, 0, :]

    actual_results = df.groupby(colmap["location"])[[colmap["dem_actual"], colmap["rep_actual"]]].first().sort_index()
    raw_forecast_data = calculate_forecast_metrics(colmap, pred_dem, pred_rep, actual_results, locations)

    raw_forecast_df = pd.DataFrame(raw_forecast_data)
    raw_forecast_df.set_index('location', inplace=True) #
    print_forecast_table(raw_forecast_df)
    ec_simulations = calculate_ev_forecast(pred_dem[..., locations != "US"], pred_rep[..., locations != "US"], raw_forecast_df, actual_results)

    forecast_path = os.path.join(results_dir, f'forecast_summary.csv')
    raw_forecast_df.to_csv(forecast_path)
    print(f"Forecast summary saved to: {forecast_path}")

    ec_simulation_path = os.path.join(results_dir, f'electoral_college_simulation.csv')
    ec_simulations.to_csv(ec_simulation_path)
    print(f"Electoral college simulation saved to: {ec_simulation_path}")

def get_args():
    psr = argparse.ArgumentParser()
    psr.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    psr.add_argument('--dataset', type=str, default='small', choices=['small', 'full', 'full_2016', 'current'], help='Dataset to use.')
    psr.add_argument('--overwrite', action='store_true', help='Overwrite existing results directory')
    psr.add_argument('--regenerate_figures', action='store_true', help='Regenerate existing figures')
    return psr.parse_args()

def load_dataset(dataset_name: str) -> tuple[PollDataset, dict]:
    """
    Load a polling dataset and its corresponding column mapping based on the dataset name.

    This function loads polling data from various sources and standardizes it into a common format
    for use in election forecasting models. It also loads the appropriate column mapping for the dataset.

    Args:
        dataset_name (str): The name of the dataset to load. Valid options are 'small', 'full', 'full_2016', and 'current'.

    Returns:
        tuple[PollDataset, dict]: A tuple containing:
            - An instance of a PollDataset subclass with the loaded polling data.
            - A dictionary with the column mapping for the dataset.

    Raises:
        ValueError: If an invalid dataset name is provided.

    """
    with open("./dataset_config.yml", 'r') as colmap_file:
        colmaps = yaml.load(colmap_file)
        colmap = colmaps["column_mappings"][dataset_name]
    if dataset_name == "small":
        polling_data = PollDataset() # automatically loads the Silver dataset. In the future allow configuration
    elif dataset_name == "full":
        polling_data = FlatPollDataset()
    elif dataset_name == "full_2016":
        polling_data = FiveThirtyEightPollDataset()
    elif dataset_name == "current":
        polling_data = FlatPollDataset(path="./data/president_polls.csv")
    else:
        raise ValueError(f"Invalid dataset: {dataset_name}")
    return polling_data, colmap

def load_or_generate_trace(
    colmap: dict,
    trace_filename: str,
    summary_path: str,
    overwrite: bool,
    year: int,
    model_name: str,
    polling_data: PollDataset,
    forecast_kwargs: dict,
    pymc_config: dict,
    model_kwargs: dict
) -> tuple[models.ElectionForecaster, az.InferenceData, pd.DataFrame]:
    """
    Load an existing trace or generate a new one if it doesn't exist or overwrite is True.

    Args:
        colmap (dict): Column mapping for the dataset.
        trace_filename (str): Path to save or load the trace file.
        summary_path (str): Path to save or load the summary file.
        overwrite (bool): Whether to overwrite existing trace and summary files.
        year (int): Election year.
        model_name (str): Name of the model class to use.
        polling_data (PollDataset): Dataset containing polling information.
        forecast_kwargs (dict): Keyword arguments for the forecaster.
        pymc_config (dict): Configuration for PyMC.
        model_kwargs (dict): Additional keyword arguments for the model.

    Returns:
        tuple: A tuple containing:
            - forecaster (ModelClass): An instance of the forecasting model.
            - trace (az.InferenceData): The PyMC trace.
            - summary (pd.DataFrame): Summary statistics of the trace.
    """
    ModelClass = getattr(models, model_name)
    forecaster = ModelClass(year, colmap, **forecast_kwargs)
    if not os.path.exists(trace_filename) or overwrite:
        _, trace, summary = forecaster.fit(
            polling_data,
            **pymc_config, 
            **model_kwargs)
        os.makedirs(os.path.dirname(trace_filename), exist_ok=True)
        print("Saving fit summary data to", summary_path)
        summary.to_csv(summary_path)

        print("Saving PyMC trace to", trace_filename)
        trace.to_netcdf(trace_filename)
        print(f"Trace saved to: {trace_filename}")
    else:
        print(f"Existing trace found. Loading trace data...\nTrace: {trace_filename}\nSummary: {summary_path}")
        trace = az.from_netcdf(trace_filename)
        summary = pd.read_csv(summary_path, index_col=0)
    return forecaster, trace, summary

def generate_state_forecast_figure(colmap, state, figure_path, trace, polling_data, forecast_kwargs, forecast_horizon, year, time_window, regenerate_figures, verbose=False):
    if os.path.exists(figure_path) and not regenerate_figures:
        if verbose:
            print(f"Figure for state {state} already exists at {figure_path}. Skipping...")
        return
    print(f"Generating figure for state: {state} (to be saved to: {figure_path})")
    data = preprocess_for_plotting(
        trace,
        polling_data,
        colmap,
        forecast_horizon,
        year,
        state,
        **forecast_kwargs,
    )
    forecast_fig, lgd = plot_forecast(colmap, *data)
    forecast_fig.savefig(figure_path, bbox_inches="tight", bbox_extra_artists=(lgd,))
    plt.close(forecast_fig)  # Close the figure after saving


if __name__ == '__main__':
    args = get_args()

    yaml = YAML(typ='safe')
    with open(args.config, 'r') as config_file:
        config = yaml.load(config_file)


    results_dir = os.path.join("./results", config["name"])
    year = config["year"]
    forecast_horizon_days = config['model_kwargs']['forecast_horizon'] * config['forecast_kwargs']['time_window_days']
    trace_filename = os.path.join(results_dir, f"pymc_trace_{year}_horizon_{forecast_horizon_days}d.nc")
    figure_dir = os.path.join(results_dir, "figures")
    figure_paths = [os.path.join(figure_dir, f"{state}_{year}_forecast.pdf") for state in VALID_LOCATIONS]
    summary_path = os.path.join(results_dir, "fit_summary.csv")

    polling_data, colmap = load_dataset(args.dataset)
    forecaster, trace, summary = load_or_generate_trace(
        colmap,
        trace_filename,
        summary_path,
        args.overwrite,
        config["year"],
        config["model_name"],
        polling_data,
        config["forecast_kwargs"],
        config["pymc"],
        config["model_kwargs"]
    )
    os.makedirs(figure_dir, exist_ok=True)
    for state, figure_path in sorted(zip(VALID_LOCATIONS, figure_paths), key=op.itemgetter(0)):
        generate_state_forecast_figure(
            colmap,
            state,
            figure_path,
            trace,
            polling_data,
            config["forecast_kwargs"],
            config["model_kwargs"]["forecast_horizon"],
            config["year"],
            forecaster.time_window_days,
            args.regenerate_figures
        )

    evaluate_forecast(trace, colmap, polling_data.filter_polls(year=year), results_dir)
