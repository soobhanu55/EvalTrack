# Gate simulation: fixed threshold vs significance test

2000 simulated pairs of runs per row. 'No change' rows compare two runs of the same scorer (any alarm is a false alarm); 'regressed' rows lower every item's success probability to 90% (about a 6-8 point drop).

| Eval size | Scenario | Fixed-threshold gate fails | Significance gate fails |
|---|---|---|---|
| 50 | no change | 24.0% | 1.0% |
| 50 | regressed (x0.90) | 66.2% | 12.9% |
| 100 | no change | 15.2% | 1.6% |
| 100 | regressed (x0.90) | 71.5% | 21.5% |
| 300 | no change | 4.2% | 2.1% |
| 300 | regressed (x0.90) | 81.5% | 64.3% |

On small evals a fixed 5-point threshold fires on noise alone; requiring the drop to be statistically real removes those false alarms, at the cost of missing some real regressions when n is small (power rises with eval size).
