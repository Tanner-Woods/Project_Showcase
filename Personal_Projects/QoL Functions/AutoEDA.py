## Automatic and generalized exploratory data analysis
## Input: n x k array
## 
## AutoEDA produces the following:
##
##      1. Distribution characteristics
##          a) (Optional) Simple summary statistics
##          b) Histogram with overlaid ECDF and estimated CDF curves
##          c) Quantile-quantile visualization
##          d) Theoretical distribution identifier
##
##      2. (Optional) Missing value wrangling
##
##      3. Data space transformation
##          a) Separate data space into qualitative and quantitative sets
##          b) Split quantitative set into discrete and continuous subsets
##
##      4. Regression assumption fulfillment tracker --> Original, NaN-wrangled, transformed
##          a) 
##          b) 
##          c) 
##          d) Homoscedasticity
##          e) Multicollinearity
##          

import numpy as np
import scipy as sp
import seaborn as sns

## Main call


## Distribution characteristics
#### Simple summary statistics
def ComplexSummaryStatistics(data):
    summary_labels = ['Skew','Kurtsosis']
    summary_array = np.zeros(data.shape)

    for i in range(0, data.shape[1]):
        values = data[:,i]

        values_count = len(values)
        values_max = 
        values_min = 
        values_mean = np.sum(values)/values_count
        values_median = 
        values_stdev = np.std(values)
        values_skew = 
        values_kurtosis = np.mean()
        values_pct_25 = 
        values_pct_50 = value_median
        values_pct_75 = 

    summary_matrix = 0
    return summary_matrix

#### Histogram with overlaid ECDF and estimated CDF curves
def HistogramDensityOverlay():


    histogram_overlaid = 0
    return histogram_overlaid

#### Quantile-quantile visualization
def QuantileQuantilePlot():


    qq_plot = 0
    return qq_plot

#### Theoretical distribution identifier
def TheoreticalDistIdentifier():
    possible_theoretical_distributions = list()

    
    return possible_theoretical_distributions

## Missing value wrangling
def NaN_Wrangler(data, input_simple_impute, input_multiple_impute):
    flag_simple_impute = input_simple_impute
    flag_multiple_impute = input_multiple_impute

    ## Checker for more than 50% missing values by parameter
    flag_too_much_missing = np.zeros(data.shape[1])
    for i in range(0, data.shape[1]):




    wrangled_matrix = 0
    return wrangled_matrix


## Data space transformation
## Separate data space into qualitative and quantitative sets

## Split quantitative set into discrete and continuous subsets




