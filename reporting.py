import numpy as np
import pandas as pd
from rich import print as rprint
from rich.markup import escape
from rich.text import Text

from utils import VALID_CONGRESSIONAL_DISTRICTS, VALID_US_STATES, ELECTORAL_VOTES_2010, EV_TO_WIN, TOTAL_EV

def calculate_forecast_metrics(
    colmap: dict,
    pred_dem: np.ndarray,
    pred_rep: np.ndarray,
    actual_results: pd.DataFrame,
    locations: np.ndarray
) -> dict:
    # Calculate mean and standard error for each location
    pred_dem_mean = pred_dem.mean(axis=(0, 1))
    pred_dem_se = pred_dem.std(axis=(0, 1)) / (pred_dem.shape[0] * pred_dem.shape[1])**0.5
    pred_rep_mean = pred_rep.mean(axis=(0, 1))
    pred_rep_se = pred_rep.std(axis=(0, 1)) / (pred_rep.shape[0] * pred_rep.shape[1])**0.5
    dem_win_prob = (pred_dem > pred_rep).mean(axis=(0, 1))

    """# Calculate mean and standard error for national prediction
    pred_dem_national_mean = pred_dem_national.mean(axis=(0, 1))
    pred_dem_national_se = pred_dem_national.std(axis=(0, 1)) / (pred_dem_national.shape[0] * pred_dem_national.shape[1])**0.5
    pred_rep_national_mean = pred_rep_national.mean(axis=(0, 1))
    pred_rep_national_se = pred_rep_national.std(axis=(0, 1)) / (pred_rep_national.shape[0] * pred_rep_national.shape[1])**0.5
    dem_national_win_prob = (pred_dem_national > pred_rep_national).mean(axis=(0, 1))

    full_dem_preds = pd.concat([pred_dem_mean.to_pandas(), pd.Series(pred_dem_national_mean.values, index=["US"])], axis=0)
    full_rep_preds = pd.concat([pred_rep_mean.to_pandas(), pd.Series(pred_rep_national_mean.values, index=["US"])], axis=0)
    full_dem_se = pd.concat([pred_dem_se.to_pandas(), pd.Series(pred_dem_national_se.values, index=["US"])], axis=0)
    full_rep_se = pd.concat([pred_rep_se.to_pandas(), pd.Series(pred_rep_national_se.values, index=["US"])], axis=0)
    full_dem_win_prob = pd.concat([dem_win_prob.to_pandas(), pd.Series(dem_national_win_prob.values, index=["US"])], axis=0)"""
    
    # Compute signed error for each location
    dem_signed_error = actual_results[colmap["dem_actual"]] - pred_dem_mean * 100
    rep_signed_error = actual_results[colmap["rep_actual"]] - pred_rep_mean * 100

    dem_winners = (actual_results[colmap["dem_actual"]] > actual_results[colmap["rep_actual"]])
    forecast_dem_winners = pd.Series((pred_dem_mean > pred_rep_mean), index=dem_winners.index)

    raw_forecast_data = {
        'location': locations,
        'd_pred_mean': pred_dem_mean * 100,
        'd_pred_se': pred_dem_se * 100,
        'r_pred_mean': pred_rep_mean * 100,
        'r_pred_se': pred_rep_se * 100,
        'margin': (pred_dem_mean - pred_rep_mean) * 100,
        'margin_se': np.sqrt(pred_dem_se**2 + pred_rep_se**2) * 100,
        'd_win_prob': dem_win_prob * 100,
        'r_win_prob': (1 - dem_win_prob) * 100,
        'd_actual': actual_results[colmap["dem_actual"]].values,
        'd_error': dem_signed_error.values,
        'r_actual': actual_results[colmap["rep_actual"]].values,
        'r_error': rep_signed_error.values,
        'winner_pred': np.where(forecast_dem_winners, "D", "R"),
        'winner_actual': np.where(dem_winners, "D", "R"),
        'call_correct': (dem_winners == forecast_dem_winners),
    }
    return raw_forecast_data

def prob_color(prob):
    if prob > 50:
        r = int(255 * (100 - prob) / 50)
        g = int(255 * (100 - prob) / 50)
        b = 255
    else:
        r = 255
        g = int(255 * prob / 50)
        b = int(255 * prob / 50)
    return f"rgb({r},{g},{b})"

def get_rating(prob):
    if prob >= 95:
        return "Safe D"
    elif prob >= 80:
        return "Likely D"
    elif prob >= 65:
        return "Lean D"
    elif prob > 35:
        return "Toss-Up"
    elif prob > 20:
        return "Lean R"
    elif prob > 5:
        return "Likely R"
    else:
        return "Safe R"
    
def error_color(error):
    if error > 10:
        return "bold red"
    elif error > 6:
        return "red"
    elif error > 3:
        return "yellow"
    else:
        return "green"

def stylize_forecast_row(state, row):
    state_text = Text(f"{state:<6}")
    state_text.stylize("bold green" if row['call_correct'] else "bold red")

    d_win_prob_text = Text(f"{row['d_win_prob']:9.1f}%")
    r_win_prob_text = Text(f"{row['r_win_prob']:9.1f}%")
    
    prob_style = prob_color(row['d_win_prob'])
    d_win_prob_text.stylize(prob_style)
    r_win_prob_text.stylize(prob_style)

    d_pred_mean_text = Text(f" {row['d_pred_mean']:6.1f}%")
    d_pred_se_text = Text(f"({row['d_pred_se']:4.1f}%)")
    prob_style = prob_color(row['d_pred_mean'])
    
    r_pred_mean_text = Text(f" {row['r_pred_mean']:6.1f}%")
    r_pred_se_text = Text(f"({row['r_pred_se']:4.1f}%)")
    
    d_pred_mean_text.stylize(prob_style)
    d_pred_se_text.stylize(prob_style)
    r_pred_mean_text.stylize(prob_style)
    r_pred_se_text.stylize(prob_style)

    formatted_row = (
        d_pred_mean_text + " " + d_pred_se_text + " " + r_pred_mean_text + " " + r_pred_se_text + " " +
        f"{row['margin']:+6.1f}% ({row['margin_se']:4.1f}%) "
    )


    rating = get_rating(row['d_win_prob'])
    rating_text = Text(f"{rating:>10}")
    rating_style = prob_color(row['d_win_prob'])
    rating_text.stylize(rating_style)

    d_actual_text = Text(f"{row['d_actual']:7.1f}%")
    r_actual_text = Text(f"{row['r_actual']:7.1f}%")
    prob_style = prob_color(row['d_actual'])
    d_actual_text.stylize(prob_style)
    r_actual_text.stylize(prob_style)

    d_error_text = Text(f"{row['d_error']:+7.1f}%")
    d_error_text.stylize(error_color(abs(row['d_error'])))
    r_error_text = Text(f"{row['r_error']:+7.1f}%")
    r_error_text.stylize(error_color(abs(row['r_error'])))

    actual_results = d_actual_text + " " + d_error_text + " " + r_actual_text + " " + r_error_text

    return state_text + formatted_row + d_win_prob_text + " " + r_win_prob_text + " " + rating_text + " " + actual_results

def print_forecast_table(raw_forecast_df):
    print(f"\n{'State':<6} {'D Pred (SE)':>15} {'R Pred (SE)':>15} {'Margin (SE)':>15} {'D Win Prob':>11} {'R Win Prob':>11} {'Rating':>10} {'D Actual':>8} {'D Error':>8} {'R Actual':>8} {'R Error':>8}")
    print("-" * 130)
    for state, row in raw_forecast_df.iterrows():
        if state == "US": 
            continue
        rprint(stylize_forecast_row(state, row))
    print("-" * 130)

    print()
    print(f"NATIONAL POPULAR VOTE (predicted):")
    rprint(stylize_forecast_row("US", raw_forecast_df.loc["US"]))
    print()

def calculate_ev_forecast(dem_no_us, rep_no_us, raw_forecast_df, actual_results):
    # Sort ELECTORAL_VOTES_2010 alphabetically to match the order of locations
    sorted_electoral_votes = dict(sorted(ELECTORAL_VOTES_2010.items()))
    ev_array = np.array(list(sorted_electoral_votes.values())) 

    dem_ev = (dem_no_us > rep_no_us).astype(int) * ev_array
    rep_ev = (rep_no_us >= dem_no_us).astype(int) * ev_array
    ec_simulations = pd.DataFrame(
        np.where((dem_no_us > rep_no_us).values.reshape(-1, dem_no_us.shape[-1]), "D", "R"),
        columns=dem_no_us.coords["location"].values
    )

    dem_total_ev = dem_ev.sum(axis=-1).values.ravel()
    rep_total_ev = rep_ev.sum(axis=-1).values.ravel()
    dem_ev_mean = dem_total_ev.mean()
    dem_ev_se = dem_total_ev.std() / np.sqrt(dem_total_ev.size)
    rep_ev_mean = rep_total_ev.mean()
    rep_ev_se = rep_total_ev.std() / np.sqrt(rep_total_ev.size)

    # Calculate probability of Democratic win
    dem_win_prob = (dem_total_ev > EV_TO_WIN).mean()
    
    state_mask = actual_results.index != 'US'
    actual_dem_ev = (raw_forecast_df.loc[state_mask, "winner_actual"] == "D").dot(ev_array)
    actual_rep_ev = TOTAL_EV - actual_dem_ev
    # Compute error in electoral vote forecast
    ev_error = abs(dem_ev_mean - actual_dem_ev)

    print("NATIONAL ELECTORAL VOTE FORECAST")
    print(f"EV Forecast: (D) {dem_ev_mean:.1f} (±{dem_ev_se:4.1f}) | (R) {rep_ev_mean:.1f} (±{rep_ev_se:4.1f}) | (D) Win prob: {dem_win_prob*100:.1f}%")
    print(f"Actual EV:   (D) {actual_dem_ev:<4}          | (R) {actual_rep_ev:<4}          |     EV Error: {ev_error:.1f}")
    print("-" * 100)

    regular_state_calls = raw_forecast_df["call_correct"].loc[raw_forecast_df.index.isin(VALID_US_STATES)].sum()
    cd_calls = raw_forecast_df["call_correct"].loc[raw_forecast_df.index.isin(VALID_CONGRESSIONAL_DISTRICTS)].sum()
    total_regular_states = len(VALID_US_STATES)
    total_cds = len(VALID_CONGRESSIONAL_DISTRICTS)
    print(f"States correct: {regular_state_calls}/{total_regular_states} ({regular_state_calls/total_regular_states*100:.1f}%); {regular_state_calls + cd_calls}/{total_regular_states + total_cds} ({(regular_state_calls + cd_calls)/(total_regular_states + total_cds)*100:.1f}%, incl. CDs)")
    print()
    return ec_simulations