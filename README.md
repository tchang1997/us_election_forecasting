# U.S. Election Forecasting

This is a hobby project for electoral forecasting! Don't take it too seriously, I have very little idea what I'm doing. 

**Objective:** predict national and state-by-state vote margins for the 2024 U.S. Presidential Election by aggregating polling data, outputting an electoral college and popular vote forecast.

### Development roadmap

**Higher priority**
* Add full forecasts for ~2020~ and 2016 (monthly up to 6 months out, then weekly for final month)
* Pollster-specific effects (*e.g.*, house-effects, RV vs. LV)
* Add "polls-plus" capability (*e.g.*, "fundamentals" like incumbency effects, economic indices, demographics via linkage to census data)

**Moderate priority**
* Add tipping point states to the report
* Add ability to weight polls/pollsters differently by quality/time (note that this is distinct from a house-effect adjustment, which is tantamount to adjusting the polllster's "meann")
* Different initial variances by state (*e.g.*, prior_precision as a vector)
* ~Different variance priors (*e.g.*, half-Cauchy?)~

**Low priority**
* Convention bounce adjustment
* A dashboard for exploring different forecasts
* Third-party adjustment (for switching to polls that have third parties)

## How to run stuff

To replicate our results (a forecast of the 2016 election), run
```
    python main.py --config config/models/polls_only.yml
```
where you can replace `--config` with a configuration file of your choice. See `config/models/polls_only.yml` for an example. Results CSVs and figures will appear in `results/`, as well as a printout of summary forecast data.

## Dataset

For initial testing, we use the Silver Bulletin dataset.The dataset contains 3,004 general election polls (among polls for other races) from the 2000-2020 election cycles, inclusive. For other elections, we use the following:

* **2024**: [FiveThirtyEight 2020 Election Forecast Polling Data, under "Presidential General Election" (current cycle)](https://projects.fivethirtyeight.com/polls/president-general/2024/national/)
* **2020**: [FiveThirtyEight 2020 Election Forecast Polling Data, under "Presidential General Election" (past cycles)](https://projects.fivethirtyeight.com/polls/president-general/2024/national/)
* **2016**: [FiveThirtyEight 2016 Election Forecast Polling Data](https://projects.fivethirtyeight.com/2016-election-forecast/)
* **Small-scale model testing (2000-2020)**: [Silver Bulletin Pollster Data](https://www.natesilver.net/p/pollster-ratings-silver-bulletin)

## Model specification

Our model is largely based on Drew Linzer's Votamatic model, described in ["Dynamic Bayesian Forecasting of Presidential Elections in the States" (2013)](https://votamatic.org/wp-content/uploads/2013/07/Linzer-JASA13.pdf), with a few "because I can" style modifications. Essentially, this model places a multivariate Gaussian random walk prior on a "state-level" and "national-level" effect variable. Notably, we don't constrain the covariance to be diagonal, which naturally allows for modeling correlations between states (but runs some risks -- that's a lot of parameters). We then combine the state and national-level effects to output probabilistic forecasts of state and national voting averages at each timepoint.

## Forecast validation

All models are fitted on polling data sourced via 538 archives. Model selection for 2024 is done based on results for the 2020 and 2016 elections based on the average number of correct states, which forgives larger polling errors in safe states, followed by the electoral vote forecast error. Metrics for each model are averaged over all forecasts for a single model. For faster iteration, for initial model selection, we only sample 2000 samples (except for the polls-only approach, when I didn't know better). 

Once model selection is finished, I'll release some 2024 forecasts.

### 2020 Election

As a reference, the FiveThirtyEight forecast predicted a 52.9% - 45.9% margin one month out with a 333 - 205 EV victory. 

|Model|(D) Win Prob.|D EV Forecast|D EV Actual|R EV Forecast|R EV Actual|States Correct (incl. DC)|
|----|----|----|----|----|----|----|
|Polls-only, one month out|91.1%|318.9|306|219.1|232|48|

### 2016 Election

Coming soon!

### 2012 Election and older

If anyone knows a source for polling data in 2012 and earlier (the Silver Bulletin dataset seems to only include general election polls starting in October) -- please reach out!

## Contact 

Email: `ctrenton` at `umich` dot `edu`.

## Old Forecast results

Initial debugging was done on a smaller dataset of polls collected from October onward.

More to be added as runs finish!

### 2016 Election 

|Model|(D) Win Prob.|D EV Forecast|D EV Actual|R EV Forecast|R EV Actual|States Correct (incl. DC)|
|----|----|----|----|----|----|----|
|Polls-only, one week out|73.5%|290.9 (± 0.3)|232|247.1 (± 0.3)|306|46|

**Disclaimer:** Note that I really didn't spend much time tuning this/doing so in any systematic manner.

States that were incorrect in our simplest polls-only model were FL, IA, ME-2, MI, PA, and WI. In particular, we basically re-created the big misses in MI, PA, and WI in the 2016 election in many polls-based forecasts. In addition, one week before the election, the [538 forecast](https://projects.fivethirtyeight.com/2016-election-forecast/) projected a (D) win probability of 75.2%. Ours is 73.5%, which is subjectively pretty close.

