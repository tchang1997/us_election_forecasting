# U.S. Election Forecasting

This is a hobby project for electoral forecasting, using [publicly-available polling data](https://www.natesilver.net/p/pollster-ratings-silver-bulletin) made available via the [Silver Bulletin](https://www.natesilver.net/) blog. Don't take it too seriously, I have very little idea what I'm doing. 

**Objective:** predict national and state-by-state vote margins for the 2024 U.S. Presidential Election by aggregating polling data, outputting an electoral college and popular vote forecast.

## How to run stuff

To replicate our results (a forecast of the 2016 election), run
```
    python main.py --config config/models/polls_only.yml
```
where you can replace `--config` with a configuration file of your choice. See `config/models/polls_only.yml` for an example. Results CSVs and figures will appear in `results/`, as well as a printout of summary forecast data.

## Dataset

The dataset contains 3,004 general election polls (among polls for other races) from the 2000-2020 election cycles, inclusive. More polling datasets will be added in the future.

## Model specification

Our model is largely based on Drew Linzer's Votamatic model, described in ["Dynamic Bayesian Forecasting of Presidential Elections in the States" (2013)](https://votamatic.org/wp-content/uploads/2013/07/Linzer-JASA13.pdf), with a few modifications. Essentially, this model places a multivariate Gaussian random walk prior on the voting averages of all states, moving *backward* in time from an initial distribution centered on an initial forecast based on election results from previous cycles. We assume state and national level effects ONLY (this is a very simple model).

## Forecast results

To be added as runs finish!

