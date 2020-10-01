import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

color_palette = sns.color_palette('Set2')

def create_event_bins(df, num_bins):
    '''
        Create bins by event size.
    '''
    min_sp = df['event_size'].min()
    max_sp = df['event_size'].max()
    bin_size = int(np.ceil((max_sp-min_sp) / num_bins))
    bin_range = np.arange(min_sp,max_sp+bin_size-1,bin_size)
    return bin_range  

def plot_precision_values(precision, ax, color):
    '''
        Plot of precision values (to the given axis), that are
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

    # set labels and title
    ax.set( xlabel=common_xlabel,
            ylabel=common_ylabel,
            title=common_title)

def plot_runtime_values(runtime, ax, color):
    '''
        Create plot of algorithm runtime values of the CPU and
        GPU implementation on the given axis.
    '''
    common_xlabel ='number of space points (thousand)'
    common_ylabel='time (s)'
    common_title='Runtime of given algorithm'

    # assume second column holds runtime values
    runtime_colname = runtime.columns[1]

    sns.scatterplot(x='event_size',
                    y=runtime_colname,
                    data=runtime,
                    ax=ax,
                    color=color_palette[color])

    # set labels and title
    ax.set( xlabel=common_xlabel,
            ylabel=common_ylabel,
            title=common_title)
      
def plot_speedup_values(speedup, ax, color):
    '''
        Plot speed-up values to given axis.
    '''
    common_xlabel='number of space points (thousand)'
    common_ylabel='speed-up factor'
    common_title='Speed-up of given algorithm'

    sns.scatterplot(x='event_size',
                    y='speedup',
                    data=speedup,
                    ax=ax,
                    color=color_palette[color])

    ax.set( xlabel=common_xlabel,
            ylabel=common_ylabel,
            title=common_title)

def sycl_seeding_plots(df_sycl):
    '''
        Create summary plot of runtime, speed-up and precision.
    '''
    # filter necessary columns for different plots
    gpu_runtime = df_sycl[['event_size', 'gpu_time']]
    cpu_runtime = df_sycl[['event_size', 'cpu_time']]
    speedup = df_sycl[['event_size', 'speedup']]
    df_sycl['precision'] = df_sycl['seed_matches'] / df_sycl['cpu_seeds'] *100
    precision = df_sycl[['event_size', 'precision']]

    # add color codes of Set2 palette
    cpu_color_runtime = 1
    gpu_color_runtime = 0
    speedup_color = 2
    precision_color = 6

    _,(runtime_axis, speedup_axis, precision_axis) = plt.subplots(1,3,figsize=(20,5))
    # override some of the default titles
    speedup_title='Speed-up factor of SYCL (GPU) seed finding algorithm'
    runtime_title='CPU and SYCL (GPU) algorithm runtime'

    # do the actual plots
    plot_runtime_values(gpu_runtime, runtime_axis, gpu_color_runtime)
    plot_runtime_values(cpu_runtime, runtime_axis, cpu_color_runtime)
    plot_speedup_values(speedup, speedup_axis, speedup_color)
    plot_precision_values(precision, precision_axis, precision_color)

    # set titles and legends
    runtime_axis.set(title=runtime_title)
    speedup_axis.set(title=speedup_title)
    # custom legend for runtime plot
    custom_legend_lines = [ Line2D([0], [0], color=color_palette[cpu_color_runtime], lw=4),
                            Line2D([0], [0], color=color_palette[gpu_color_runtime], lw=4)]
    runtime_axis.legend(custom_legend_lines, ['CPU','GPU'], title='Algorithm')

def sycl_cuda_comparison_plot(df_sycl, df_cuda):
    '''
        Compare results of not entirely same SYCL and CUDA seeding algorithms.
    '''
    speedup_sycl = df_sycl[['event_size', 'speedup']]
    speedup_cuda = df_cuda[['event_size', 'speedup']]

    runtime_sycl = df_sycl[['event_size', 'gpu_time']]
    runtime_cuda = df_cuda[['event_size', 'gpu_time']]
    runtime_cpu = df_sycl[['event_size', 'cpu_time']]

    # add color codes
    cuda_color=6
    sycl_color=2
    cpu_color=1

    _,(compare_runtime_axis, compare_speedup_axis) = plt.subplots(1,2,figsize=(15,7))
    # override default titles
    speedup_title='Speed-up factor of SYCL and CUDA seed finding algorithms'
    runtime_title='Runtime of SYCL, CUDA and CPU seed finding algorithms'

    plot_runtime_values(runtime_cuda, compare_runtime_axis, cuda_color)
    plot_runtime_values(runtime_sycl, compare_runtime_axis, sycl_color)
    plot_runtime_values(runtime_cpu, compare_runtime_axis, cpu_color)

    plot_speedup_values(speedup_cuda, compare_speedup_axis, cuda_color)
    plot_speedup_values(speedup_sycl, compare_speedup_axis, sycl_color)

    # set custom titles
    compare_speedup_axis.set(title=speedup_title)
    compare_runtime_axis.set(title=runtime_title)
    # add custom legends
    custom_legend_lines = [ Line2D([0], [0], color=color_palette[cuda_color], lw=4),
                            Line2D([0], [0], color=color_palette[sycl_color], lw=4),
                            Line2D([0], [0], color=color_palette[cpu_color], lw=4)]
    compare_runtime_axis.legend(custom_legend_lines, ['CUDA','SYCL', 'CPU'], title='Algorithm')
    compare_speedup_axis.legend(custom_legend_lines[:2], ['CUDA','SYCL'], title='Algorithm')

def plot_mean_with_error(df, bin_range, ax):
    '''
        Group values by event size and plot their mean with the standard error.
    '''
    l = len(bin_range)
    bin_of_event = pd.cut(df['event_size'],bin_range)
    df = df.assign(bins=bin_of_event)
    df = df.drop(columns=['event_size'])
    grouped = df.groupby('bins')
    mean = grouped.mean().iloc[:,0].to_list()
    std = grouped.std().iloc[:,0].to_list()
    error_lower = list(np.subtract(mean,std))
    error_higher = list(np.add(mean,std))
    avg_bins = (bin_range[:l-1] + bin_range[1:]) / 2
    sns.lineplot(avg_bins, mean, ax=ax)
    ax.fill_between(avg_bins, error_lower, error_higher, facecolor='lightblue')

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
