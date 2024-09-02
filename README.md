# U.S. Election Forecasting

This is a hobby project for electoral forecasting! Don't take it too seriously, I have very little idea what I'm doing. 

**Objective:** predict national and state-by-state vote margins for the 2024 U.S. Presidential Election by aggregating polling data, outputting an electoral college and popular vote forecast.

### Development roadmap

**Higher priority**
* Add results for monthly forecasts up to 6 months out, back-casting 2 months, then weekly for final month
* Pollster-specific effects (*e.g.*, house-effects, RV vs. LV)
* Add "polls-plus" capability (*e.g.*, "fundamentals" like incumbency effects, economic indices, demographics via linkage to census data)

**Moderate priority**
* Add tipping point states to the report
* Add ability to weight polls/pollsters differently by quality/time (note that this is distinct from a house-effect adjustment, which is tantamount to adjusting the polllster's "mean")
* Different initial variances by state (*e.g.*, prior_precision as a vector) -- to model state elasticity

**Low priority**
* Convention bounce adjustment
* A dashboard for exploring different forecasts
* Third-party adjustment (for switching to polls that have third parties) -- w/o controlling for Gary Johnson, some of the 2016 forecasts look visually wonky.

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

By default, we only consider polls up to two months back from the forecast horizon (*e.g.*, we use August and September polls for a "start of October" forecast). 

## Predictions

We output three separate forecats from the same posterior: voting percentages, win probabilities, and an electoral vote forecast.

**Voting percentages.** To output mean voting percentages, we take the mean of the estimated state-/national-level vote parameter for each party (a proportion from 0 to 1) over all election simulations.

**Win probabilities.** We take the proportion of times that each party's vote parameter exceeds that of the other over all election simulations.

**Electoral vote forecast.** For each sample, we compute the mean number of electoral votes won by each candidate per election simulation, and output the mean. Note that this forecast is not guaranteed to match the voting percentage forecast due to the all-or-nothing nature of the electoral college and correlated errors across states in our model.


## Forecast validation

All models are fitted on polling data sourced via 538 archives. Model selection for 2024 is done based on results for the 2020 and 2016 elections based on the average number of correct states (incl. DC), which forgives larger polling errors in safe states, followed by the electoral vote forecast error, then popular vote error (absolute DEM + REP error). Metrics for each model are averaged over all forecasts for a single model. For faster iteration, for initial model selection, we only sample 2000 samples (4 chains, 500 each, 1000 tuning steps no matter what).

Since the precision parameter (variance around the final estimate) should also decrease as we get closer to the election, after selecting a type of model (*e.g.,* polls-only vs. polls-plus), we validate schedules for that parameter within the chosen model class.

Once model selection is finished, I'll release some 2024 forecasts.

### 2020 Election

As a reference, the FiveThirtyEight forecast predicted a 52.9% - 45.9% margin one month out with a 333 - 205 EV victory. 

|Model|(D) Win Prob. (PV)|D PV Forecast|D PV Actual|R PV Forecast|R PV Actual|(D) Win Prob. (EV)|D EV Forecast|D EV Actual|R EV Forecast|R EV Actual|States Correct (incl. DC)|EV Error|PV Error
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
|Polls-only, one month out|84.5%|49.6%|51.3%|45.9%|46.8%|83.3%|305.8|306|232.2|232|**47**|**0.2**|**2.5%**|
|Polls-only, two months out|82.2%|49.8%|51.3%|45.5%|46.8%|72.4%|293.5|306|244.5|232|**49**|**12.5**|**2.9%**|
|Polls-only, three months out|66.0%|50.1%|51.3%|46.9%|46.8%|66.8%|286.6|306|251.4|232|**49**|**19.4**|**1.2%%**|


### 2016 Election

As a reference, the FiveThirtyEight forecast predicted a 52.9% - 45.9% margin one month out with a 329.2 - 208.7 EV victory. Note that our forecasts for 2016 are a little more wonky, since we haven't accounted for the effect of Gary Johnson in polling numbers.

|Model|(D) Win Prob. (PV)|D PV Forecast|D PV Actual|R PV Forecast|R PV Actual|(D) Win Prob. (EV)|D EV Forecast|D EV Actual|R EV Forecast|R EV Actual|States Correct (incl. DC)|EV Error|PV Error|
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
|Polls-only, one month out|72.5%|51.8%|48.0%|47.4%|45.8%|93.8%|322.7|232|215.3|306|**44**|**90.7**|**5.3%**|


### 2012 Election and older

If anyone knows a source for polling data in 2012 and earlier (the Silver Bulletin dataset seems to only include general election polls starting in October) -- please reach out!

## Contact 

Email: `ctrenton` at `umich` dot `edu`.
