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
    locations = trace.posterior.coords["location"]
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
    psr.add_argument('--dataset', type=str, default='small', choices=['small', 'full', 'full_2016'], help='Dataset to use.')
    psr.add_argument('--overwrite', action='store_true', help='Overwrite existing results directory')
    psr.add_argument('--regenerate_figures', action='store_true', help='Regenerate existing figures')
    return psr.parse_args()

def load_dataset(dataset_name):
    with open("./dataset_config.yml", 'r') as colmap_file:
        colmaps = yaml.load(colmap_file)
        colmap = colmaps["column_mappings"][dataset_name]
    if dataset_name == "small":
        polling_data = PollDataset() # automatically loads the Silver dataset. In the future allow configuration
    elif dataset_name == "full":
        polling_data = FlatPollDataset()
    elif dataset_name == "full_2016":
        polling_data = FiveThirtyEightPollDataset()
    else:
        raise ValueError(f"Invalid dataset: {dataset_name}")
    return polling_data, colmap

def load_or_generate_trace(colmap, trace_filename, summary_path, overwrite, year, model_name, polling_data, forecast_kwargs, pymc_config, model_kwargs):
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
    trace_filename = os.path.join(results_dir, f"pymc_trace_{year}_horizon_{config['model_kwargs']['forecast_horizon']}d.nc")
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
