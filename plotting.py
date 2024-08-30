
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd

from typing import Optional

plt.rcParams["font.family"] = "serif"

# demp_pi_sample = trace["posterior"]["dem_pi"]
def plot_forecast(
        dem_pi_sample,
        rep_pi_sample,
        forecast_horizon,
        header,
        state_idx,
        election_result: Optional[pd.DataFrame] = None,
        polling_data: Optional[pd.DataFrame] = None,
        ci: Optional[float] = 95.,
    ):

    dem_pi_sample = dem_pi_sample[:, :, ::-1, state_idx]
    rep_pi_sample = rep_pi_sample[:, :, ::-1, state_idx] # forward in time now

    # compute mean and CI
    dem_mean = dem_pi_sample.mean(axis=(0, 1))
    rep_mean = rep_pi_sample.mean(axis=(0, 1))
    ci_lower = (100 - ci) / 200
    ci_upper = 1 - ci_lower
    dem_ci = np.quantile(dem_pi_sample, ci_lower, axis=(0, 1)), np.quantile(dem_pi_sample, ci_upper, axis=(0, 1))
    rep_ci = np.quantile(rep_pi_sample, ci_lower, axis=(0, 1)), np.quantile(rep_pi_sample, ci_upper, axis=(0, 1))
                        
    fig, ax = plt.subplots()

    #first_row = orig_data.iloc[0]

    final_margin = dem_pi_sample[..., -1].mean() - dem_pi_sample[..., -1].mean()
    win_prob = (dem_pi_sample[..., -1] > rep_pi_sample[..., -1]).mean()

    ax.set_title(header["cand1_name"] + " (D) vs. " + header["cand2_name"] + " (R), State: " + header["location"] + " (" + header["type_simple"] + f")\nFinal margin: D{100 * final_margin:+.1f} - Dem. Win: {win_prob * 100:.2f}%")
    ax.set_ylim((30, 70))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    total_weeks = np.arange(len(dem_mean))
    forecast_weeks = total_weeks[:-forecast_horizon+1]
    predicted_weeks = total_weeks[-forecast_horizon:]

    ax.plot(forecast_weeks, dem_mean[:-forecast_horizon+1] * 100, color="blue", marker=".", label="DEM (%) [95% CI]")
    ax.plot(forecast_weeks, rep_mean[:-forecast_horizon+1] * 100, color="red", marker=".", label="REP (%) [95% CI]")
    ax.plot(predicted_weeks, dem_mean[-forecast_horizon:] * 100, color="tab:blue", marker=".", alpha=0.7, linestyle="dashed")
    ax.plot(predicted_weeks, rep_mean[-forecast_horizon:] * 100, color="tab:red", marker=".", alpha=0.7, linestyle="dashed")
    ax.fill_between(total_weeks, y1=dem_ci[0] * 100, y2=dem_ci[1] * 100, color="tab:blue", alpha=0.2)
    ax.fill_between(total_weeks, y1=rep_ci[0] * 100, y2=rep_ci[1] * 100, color="tab:red", alpha=0.2)

    ax.set_xlabel("Date")
    ax.set_ylabel("Vote share (%)")
    ax.vlines(len(dem_mean) - forecast_horizon, linestyle="dotted", color="black", ymin=0., ymax=100., label="Validation horizon")

    raw_dates = np.array([(header["electiondate"] - pd.Timedelta(days=i)) for i in range(len(dem_mean) + 1)])
    dates_list = np.vectorize(lambda x: x.strftime('%m/%d/%Y'))(raw_dates)
    dates_list[0] += "\n(Election Day)"

    ax.set_xticks(range(0, len(dates_list), 2))
    ax.set_xticklabels(dates_list[::-2], rotation=45)

    if polling_data is not None:
        # then we expect polls to be a dataframe with polldate, cand1_pct, cand2_pct
        poll_x = (polling_data["polldate"].map(lambda x: x.toordinal()) - raw_dates.min().toordinal()) / (raw_dates.max().toordinal() - raw_dates.min().toordinal()) * len(raw_dates)
        ax.scatter(poll_x, polling_data["cand1_pct"], color="blue", alpha=0.2)
        ax.scatter(poll_x, polling_data["cand2_pct"], color="red", alpha=0.2)

    if election_result is not None:
        # then, we expect an iterable of the Dem vote share and the Rep vote share
        ax.hlines(election_result[["cand1_actual", "cand2_actual"]], xmin=0, xmax=len(dem_mean), linestyles="dotted", colors=["blue", "red"])

    lgd = fig.legend(loc="lower center", ncols=3, title="Legend", bbox_to_anchor=(0.5, -0.25))
    fig.tight_layout()
    return fig, lgd
