import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def load_results(generate_result_fn, experiments=None, results_file=None, overwrite=False):
    if experiments is None and results_file is None:
        raise ValueError('Need to provide experiments or/and results_file')

    if experiments is None and not os.path.exists(results_file):
        raise ValueError('Results file does not exist')

    if overwrite or (results_file is None) or (results_file is not None and not os.path.exists(results_file)):
        if experiments is None:
            raise ValueError('Need to provide experiments')

        # Generate data
        result_dicts = []

        for exp in tqdm(experiments):
            result_dicts.append(generate_result_fn(exp))

        results = pd.DataFrame(result_dicts)

        if results_file is not None:
            directory, filename = os.path.split(results_file)
            if not os.path.exists(directory):
                os.mkdir(directory)

            pd.to_pickle(results, results_file)

    else:
        if not os.path.exists(results_file):
            raise ValueError('Results file does not exist')

        results = pd.read_pickle(results_file)

    return results


def plot_results_dict(plot_dict, xlabel, ylabel,
                      error_alpha=0.5, x_lim_factor=1.05, y_lim_factor=1.1,
                      legend_params={}, xlabel_params={}, ylabel_params={}, yticks_params=None, title_params=None,
                      linewidth=1, clip_on=False, zorder=3, budget_scaling=1, xlim=None, ylim=None, step=False,
                      keys=None):
    pal = sns.color_palette('colorblind', n_colors=len(plot_dict))

    max_budget = 0
    max_metric = 0

    if keys is None:
        keys = plot_dict.keys()

    for i, k in enumerate(keys):
        if 'label' in plot_dict[k].keys():
            label = plot_dict[k]['label']
        else:
            label = k

        budgets = plot_dict[k]['budgets'] * budget_scaling
        metrics = plot_dict[k]['cert_metrics']

        if step:
            plt.step(budgets, metrics, label=label, color=pal[i], zorder=zorder, clip_on=clip_on, linewidth=linewidth)
        else:
            plt.plot(budgets, metrics, label=label, color=pal[i], zorder=zorder, clip_on=clip_on, linewidth=linewidth)

        if 'errors' in plot_dict[k].keys():
            errors = plot_dict[k]['errors']

            if step:
                plt.fill_between(budgets, metrics-errors, metrics+errors, alpha=error_alpha, color=pal[i], zorder=zorder, clip_on=clip_on, linewidth=0, step='post')
            else:
                plt.fill_between(budgets, metrics-errors, metrics+errors, alpha=error_alpha, color=pal[i], zorder=zorder, clip_on=clip_on, linewidth=0)

        print(f'Max error for {k} is {errors.max()}')

        max_budget = max(max_budget, budgets.max())
        max_metric = max(max_metric, metrics.max())

    if xlim is not None:
        plt.xlim(0, xlim)
    elif x_lim_factor is not None:
        plt.xlim(0, x_lim_factor * max_budget)
    else:
        plt.xlim(left=0)

    if ylim is not None:
        plt.ylim(0, ylim)
    elif y_lim_factor is not None:
        plt.ylim(0, y_lim_factor * max_metric)
    else:
        plt.ylim(bottom=0)

    plt.xlabel(xlabel, **xlabel_params)
    plt.ylabel(ylabel, **ylabel_params)

    if title_params is not None:
        plt.title(**title_params)

    if yticks_params is not None:
        plt.yticks(**yticks_params)

    plt.legend(**legend_params)
