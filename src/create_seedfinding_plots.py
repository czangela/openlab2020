import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

color_palette = sns.color_palette('Set2')

def plot_precision_values(precision, ax, color):
    '''
        Create plot with two subplots of precision values, that are
        calculated the following way: matched_seeds / cpu_seeds.
         - matched_seeds is the number of seeds that both the CPU and
         GPU algorithm found
         - cpu_seeds is the number of seeds found by the CPU algorithm
    '''
    common_xlabel='number of space points (thousand)'
    common_ylabel='percentage (%)'
    common_title='Percentage of matched seeds'

    # scatter plot
    sns.scatterplot(x='event_size',
                    y='precision',
                    data=precision,
                    ax=ax,
                    color=color_palette[color])

    ax.set(xlabel=common_xlabel, ylabel=common_ylabel, title=common_title)

def plot_runtime_values(runtime, ax):
    '''
        Create plot of algorithm runtime values of the CPU and
        GPU implementation.
    '''
    common_xlabel ='number of space points (thousand)'
    common_ylabel='time (s)'
    cpu_color = color_palette[1]
    gpu_color = color_palette[0]

    sns.scatterplot(x='event_size', y='cpu_time', data=runtime, ax=ax, color=cpu_color)
    sns.scatterplot(x='event_size', y='gpu_time', data=runtime, ax=ax, color=gpu_color)

    # set labels and legends
    ax.set(xlabel=common_xlabel, ylabel=common_ylabel)
    # custom legend
    custom_legend_lines = [ Line2D([0], [0], color=cpu_color, lw=4),
                            Line2D([0], [0], color=gpu_color, lw=4)]
    ax.legend(custom_legend_lines, ['CPU','GPU'], title='Algorithm')
    
def plot_speedup_values(speedup_sycl, ax, color):
    '''
        Create plot with two subplots of speed-up values.
    '''
    common_xlabel='number of space points (thousand)'
    common_ylabel='speed-up factor'

    sns.scatterplot(x='event_size',
                    y='speedup',
                    data=speedup_sycl,
                    ax=ax,
                    color=color_palette[color])

    ax.set(xlabel=common_xlabel,
            ylabel=common_ylabel)

def sycl_seeding_plots(df_sycl):
    # filter necessary columns for different plots
    runtime = df_sycl[['event_size', 'gpu_time', 'cpu_time']]
    speedup = df_sycl[['event_size', 'speedup']]
    df_sycl['precision'] = df_sycl['seed_matches'] / df_sycl['cpu_seeds'] *100
    precision = df_sycl[['event_size', 'precision']]

    _,(runtime_axis, speedup_axis, precision_axis) = plt.subplots(1,3,figsize=(20,5))
    speedup_title='Speed-up factor of SYCL (GPU) seed finding algorithm'
    runtime_title='CPU and SYCL (GPU) algorithm runtime'

    plot_runtime_values(runtime, runtime_axis)
    plot_speedup_values(speedup, speedup_axis, 2)
    plot_precision_values(precision, precision_axis, 6)

    runtime_axis.set(title=runtime_title)
    speedup_axis.set(title=speedup_title)

    _,(single_runtime_axis) = plt.subplots(1,1, figsize=(7,7))
    plot_runtime_values(runtime,single_runtime_axis)
    single_runtime_axis.set(title=runtime_title)

    _,(single_speedup_axis) = plt.subplots(1,1, figsize=(7,7))
    plot_speedup_values(speedup,single_speedup_axis, 2)
    single_speedup_axis.set(title=speedup_title)

    _,(single_precision_axis) = plt.subplots(1,1, figsize=(7,7))
    plot_precision_values(precision,single_precision_axis,6)

def sycl_cuda_comparison_plot(df_sycl, df_cuda):
    
    speedup_sycl = df_sycl[['event_size', 'speedup']]
    speedup_cuda = df_cuda[['event_size', 'speedup']]
    cuda_color=6
    sycl_color=2

    _,(compare_axis) = plt.subplots(1,1,figsize=(7,7))
    speedup_title='Speed-up factor of SYCL and CUDA seed finding algorithm'
    plot_speedup_values(speedup_cuda, compare_axis, cuda_color)
    plot_speedup_values(speedup_sycl, compare_axis, sycl_color)
    compare_axis.set(title=speedup_title)
    custom_legend_lines = [ Line2D([0], [0], color=color_palette[cuda_color], lw=4),
                            Line2D([0], [0], color=color_palette[sycl_color], lw=4)]
    compare_axis.legend(custom_legend_lines, ['CUDA','SYCL'], title='Algorithm')

# read csv file of seedfinding results to dataframe
df_sycl = pd.read_csv('../data/seedfinding_sycl.csv')
df_cuda = pd.read_csv('../data/seedfinding_cuda.csv')

# divide event size by a thousand
df_sycl['event_size'] = df_sycl['event_size'] / 1000
df_cuda['event_size'] = df_cuda['event_size'] / 1000

# plotting
sycl_seeding_plots(df_sycl)
sycl_cuda_comparison_plot(df_sycl,df_cuda)

plt.show()
