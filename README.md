# U.S. Election Forecasting

This is a hobby project for electoral forecasting, using [publicly-available polling data](https://www.natesilver.net/p/pollster-ratings-silver-bulletin) made available via the [Silver Bulletin](https://www.natesilver.net/) blog. Don't take it too seriously, I have very little idea what I'm doing. 

**Objective:** predict national and state-by-state vote margins for the 2024 U.S. Presidential Election by aggregating polling data, outputting an electoral college and popular vote forecast.

![National Forecast](https://github.com/tchang1997/us_election_forecasting/blob/main/US_2016_forecast.pdf)

This chart shows the predicted vote share over time for the Democratic and Republican candidates, along with 95% confidence intervals. The vertical dotted line indicates the forecast horizon (1 week before the election in this case). Actual polls are shown as scattered points, while the final election result is indicated by the horizontal dotted lines.

## How to run stuff

To replicate our results (a forecast of the 2016 election), run
```
    python main.py --config config/models/polls_only.yml
```
where you can replace `--config` with a configuration file of your choice. See `config/models/polls_only.yml` for an example. Results CSVs and figures will appear in `results/`, as well as a printout of summary forecast data.

## Dataset

The dataset contains 3,004 general election polls (among polls for other races) from the 2000-2020 election cycles, inclusive. More polling datasets will be added in the future.

## Model specification

Our model is largely based on Drew Linzer's Votamatic model, described in ["Dynamic Bayesian Forecasting of Presidential Elections in the States" (2013)](https://votamatic.org/wp-content/uploads/2013/07/Linzer-JASA13.pdf), with a few "because I can" style modifications. Essentially, this model places a multivariate Gaussian random walk prior on a "state-level" and "national-level" effect variable. Notably, we don't constrain the covariance to be diagonal, which naturally allows for modeling correlations between states (but runs some risks -- that's a lot of parameters). We then combine the state and national-level effects to output probabilistic forecasts of state and national voting averages at each timepoint.

## Forecast results

More to be added as runs finish!

### 2016 Election 

|Model|(D) Win Prob.|D EV Forecast|D EV Actual|R EV Forecast|R EV Actual|States Correct (incl. DC)|
|----|----|----|----|----|----|----|
|Polls-only, one week out|73.5%|290.9 (± 0.3)|232|247.1 (± 0.3)|306|46|

**Disclaimer:** Note that I really didn't spend much time tuning this/doing so in any systematic manner.

States that were incorrect in our simplest polls-only model were FL, IA, ME-2, MI, PA, and WI. In particular, we basically re-created the big misses in MI, PA, and WI in the 2016 election in many polls-based forecasts. In addition, one week before the election, the [538 forecast](https://projects.fivethirtyeight.com/2016-election-forecast/) projected a (D) win probability of 75.2%. Ours is 73.5%, which is subjectively pretty close.

