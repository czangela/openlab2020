import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

def create_event_bins(df, num_bins):
    '''
        Create bins by event size.
    '''
    min_sp = df['event_size'].min()
    max_sp = df['event_size'].max()
    bin_size = int(np.ceil((max_sp-min_sp) / num_bins))
    bin_range = np.arange(min_sp,max_sp+bin_size-1,bin_size)
    return bin_range

def plot_precision_values(precision, bin_range):
    '''
        Create plot with two subplots of precision values, that are
        calculated the following way: matched_seeds / cpu_seeds.
         - matched_seeds is the number of seeds that both the CPU and
         gpu (SYCL) algorithm found
         - cpu_seeds is the number of seeds found by the CPU algorithm
    '''
    _, (ax0, ax1) = plt.subplots(1,2,figsize=(15,7))
    common_color = sns.color_palette('Set2')[4]
    common_xlabel='number of space points'
    common_ylabel='precision (%)'

    # scatter plot
    sns.scatterplot(x='event_size',
                    y='precision',
                    data=precision,
                    ax=ax0,
                    color=common_color)

    # binned plot
    bin_of_event = pd.cut(precision['event_size'], bin_range)
    precision = precision.assign(bins=bin_of_event)

    prec_grouped = precision.groupby('bins')
    prec_grouped.mean()[['precision']].plot(ax=ax1,
                                            style='.--',
                                            markersize=15,
                                            color=common_color)
    ax0.set(xlabel=common_xlabel, ylabel=common_ylabel, title='Percentage of matched seeds')
    ax1.set(xlabel=common_xlabel, ylabel=common_ylabel, title='Mean percentage of matched seeds of binned events')
    plt.xticks(rotation=15)

def plot_runtime_values(runtime):
    '''
        Create plot of algorithm runtime values of the CPU and SYCL
        (GPU) implementation.
    '''
    _, ax0 = plt.subplots(1,1,figsize=(8,8))
    common_xlabel ='number of space points'
    common_ylabel='time (s)'
    cpu_color = sns.color_palette('Set2')[1]
    gpu_color = sns.color_palette('Set2')[0]

    sns.scatterplot(x='event_size', y='cpu_time', data=runtime, ax=ax0, color=cpu_color)
    sns.scatterplot(x='event_size', y='gpu_time', data=runtime, ax=ax0, color=gpu_color)

    # set labels and legends
    ax0.set(xlabel=common_xlabel, ylabel=common_ylabel, title='CPU and SYCL (GPU) algorithm runtime')
    # custom legend
    custom_legend_lines = [ Line2D([0], [0], color=cpu_color, lw=4),
                            Line2D([0], [0], color=gpu_color, lw=4)]
    ax0.legend(custom_legend_lines, ['CPU','GPU'], title='Algorithm')
    
def plot_speedup_values(speedup, bin_range):
    '''
        Create plot with two subplots of speed-up values.
    '''
    _, (ax0, ax1) = plt.subplots(1,2,figsize=(15,7))
    common_xlabel='number of space points'
    common_ylabel='speed-up factor'
    common_color = sns.color_palette('Set2')[2]

    sns.scatterplot(x='event_size',
                    y='speedup',
                    data=speedup,
                    ax=ax0,
                    color=common_color)

    bin_of_event = pd.cut(speedup['event_size'],bin_range)
    speedup = speedup.assign(bins=bin_of_event)
    speedup.groupby('bins').mean()[['speedup']].plot(   ax = ax1,
                                                        style='.--',
                                                        markersize=15,
                                                        color=common_color)

    ax0.set(xlabel=common_xlabel,
            ylabel=common_ylabel,
            title='Speed-up factor of SYCL GPU seed finding algorithm')

    ax1.set(xlabel=common_xlabel,
            ylabel=common_ylabel,
            title='Mean speed-up factor of binned events')
    plt.xticks(rotation=15)

# read csv file of seedfinding results to dataframe
df = pd.read_csv('../data/seedfinding_sycl.csv')
bin_range = create_event_bins(df, 10)

# filter necessary columns for different plots
runtime = df[['event_size', 'gpu_time', 'cpu_time']]
speedup = df[['event_size', 'speedup']]
df['precision'] = df['seed_matches'] / df['cpu_seeds']
precision = df[['event_size', 'precision']]

# plotting
plot_precision_values(precision, bin_range)
plot_speedup_values(speedup, bin_range)
plot_runtime_values(runtime)
plt.show()
