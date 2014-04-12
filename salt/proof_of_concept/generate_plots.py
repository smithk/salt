import numpy as np
import sys
import os
from matplotlib import use as set_backend
from salt.evaluate.evaluate import format_p_value
set_backend('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict
import cPickle
from salt.evaluate.evaluate import get_statistics, p_stud


def setup_subplot(subplot, dataset_name, alpha):
    subplot.cla()
    subplot.axvline(0, linestyle='--', color='k', lw=1.2)
    subplot.set_title('Dataset: {0}. $\\alpha={1}$'.format(dataset_name, alpha))
    #subplot.set_title(u'{0}. $\\alpha={1}$'.format(title, alpha))
    subplot.set_xlabel('Performance index')
    subplot.set_xlim([-0.1, 1])
    subplot.yaxis.grid(False)


def create_subplot():
    #summary_figure = plt.figure(figsize=(7, 6), dpi=90)
    summary_figure = plt.figure(figsize=(5.5, 6), dpi=100)

    subplot = summary_figure.add_subplot(111)
    subplot.format_coord = lambda x, y: ''
    subplot.set_ylim([0, 1])
    subplot.xaxis.grid(False)
    subplot.set_ylabel('Score')
    return subplot


def create_pvalue_axis(subplot):
    pvalue_axis = subplot.twinx()
    pvalue_axis.format_coord = lambda x, y: ''
    pvalue_axis.set_ylim([0, 1])
    pvalue_axis.xaxis.grid(False)
    pvalue_axis.set_ylabel('p-value (with respect to the best)')
    return pvalue_axis


def display_optimized_default_chart(dataset_name, default_data, optimized_data, means, names, pvalues, pcritical=0.05):
    try:
        if True:  # self.summary_plot is None or num_learners != len(self.summary_plot.values()):
            subplot = create_subplot()
            pvalue_axis = create_pvalue_axis(subplot)
            setup_subplot(subplot, dataset_name, pcritical)
            positions = np.arange(1, len(optimized_data) + 1)
            #positions = np.vstack((positions, positions + 0.5))
            default_means = [np.mean(row) for row in default_data]
            if True:  # show defaults
                ghostcolor = '#aaaaaa'
                edgecolor = '#efefef'
                mediancolor = '#efefef'
                alpha = 0.4
                a = subplot.boxplot(default_data, positions=positions + 0.325, widths=0.25, patch_artist=True, vert=False, notch=True)
                plt.setp(a['whiskers'], color=ghostcolor, linestyle='-', alpha=alpha, linewidth=1.2)
                plt.setp(a['boxes'], color=ghostcolor, edgecolor=edgecolor, alpha=alpha)
                plt.setp(a['caps'], color=ghostcolor, alpha=alpha, linewidth=1.2)
                plt.setp(a['fliers'], color=ghostcolor, alpha=alpha, linewidth=1.2, marker='o', markersize=3.2, markeredgecolor=ghostcolor)
                plt.setp(a['medians'], color=mediancolor, linewidth=1.2, alpha=alpha)
                boxplot_width = 0.5
            #else:
            boxplot_width = 0.9
            subplot.axvline(max(means), linestyle=':', color='w', lw=1.2, zorder=0)  # Best mean
            offset = 0
            if True:  # show_defaults:
                offset = 0.15
                positions = positions - offset
            summary_plot = subplot.boxplot(optimized_data, positions=positions,
                                           widths=boxplot_width - offset * 2,
                                           patch_artist=True, vert=False,
                                           notch=True)
            yticks = subplot.get_yticks()
            yticks = np.hstack((yticks, yticks[-1] + 1))[::2]
            subplot.barh(yticks - 0.5 + offset, [1.1] * len(yticks), height=1, left=-0.1, color='#cdcdcd', linewidth=0, alpha=0.25, zorder=0)
            setup_legend(subplot)
            subplot.set_ylim([0.4, len(optimized_data) + 0.6])
            bestindex = np.nanargmax(means)
            accept = pvalues > pcritical
            setup_boxplot(subplot, summary_plot, pvalue_axis, names, pvalues, accept, bestindex)
            plot_means(subplot, means, vertical_offset=-offset)
            if True:  # show defaults
                plot_means(subplot, default_means, vertical_offset=0.325, alpha=0.4, boxheight=0.05)
            plt.tight_layout(rect=(0, 0.07, 1, 1))
            #  Significantly different from the best (95% confidence)
            plt.savefig("figures/optimized_vs_default_{0}.png".format(dataset_name))
            #plt.show(block=True)
        else:
            for i in range(len(means)):
                pass
                #self.summary_plot[i].set_height(means[i])
        #self.summary_figure.canvas.draw()
    except Exception as exc:
        print("Problem with plot: {0} on {1}".format(exc, None))


def plot_means(subplot, means, vertical_offset=0, alpha=1, boxheight=0.08):
    '''Plot the means.'''
    height = subplot.transData.transform([(0, boxheight), (0, 0)])
    height = height[0, 1] - height[1, 1]
    #self.subplot.plot(means, np.arange(1, len(means) + 1) + vertical_offset, 'ro', color='w', marker="$|$", markeredgecolor='k', markersize=8)
    subplot.plot(means, np.arange(1, len(means) + 1) + vertical_offset, 'ro', color='w', marker=[(0, -.5), (0, .5)], markeredgecolor='k', linewidth=2, markersize=height, markeredgewidth=2, alpha=alpha)
    #self.subplot.set_title('Dataset: {0}. $\\alpha$ threshold: {1}'.format(title, self.alpha.get()))


def setup_legend(subplot):
    '''Create the legend for the summary plot.'''
    artists = [plt.Line2D((0, 0), (2, 2), linewidth=0, marker=[(0, -.5), (0, .5)], markeredgecolor='k', markeredgewidth=2),
               #           markersize=5, markerfacecolor='w'),
               plt.Rectangle((0, 0), 1, 1, facecolor='#aaaaaa', edgecolor='#efefef', alpha=0.4),
               plt.Rectangle((0, 0), 1, 1, facecolor='#b7b7b7'),
               plt.Rectangle((0, 0), 1, 1, facecolor='#3387bc')]
    legend = subplot.legend(artists,
                            ['mean', 'default parameters', 'significant difference',
                            'non-significant difference'],
                            ncol=4, bbox_to_anchor=(1.15, -0.14), prop={'size': 8},
                            numpoints=1, borderaxespad=0)
    legend_frame = legend.get_frame()
    legend_frame.set_linewidth(0)
    legend_frame.set_alpha(0.75)


def setup_boxplot(subplot, summary_plot, pvalue_axis, names, pvalues, accept, bestindex):
    lw = 1.4
    plt.setp(summary_plot['whiskers'], color='#333333', linestyle='-', lw=lw)
    yticklabels = plt.getp(subplot, 'yticklabels')
    if len(accept) > 0:
        for i, box in enumerate(summary_plot['boxes']):
            if i >= len(accept):
                plt.setp(box, facecolor="#ffffff", edgecolor='#888888', lw=0.5, alpha=0.7)
                continue
            if accept[i].item():
                plt.setp(box, facecolor="#3387bc", edgecolor='#3387bc', lw=0)
            else:
                plt.setp(box, facecolor='#b0b0b0', edgecolor='#b7b7b7', lw=0)
            if yticklabels is not None:
                if not accept[i].item():
                    yticklabels[i].set_alpha(0.5)
    plt.setp(summary_plot['caps'], color='#333333', lw=lw)
    plt.setp(summary_plot['fliers'], color='#333333', markeredgecolor='#333333', marker='o', markersize=3.2)
    plt.setp(summary_plot['medians'], color='#e5e5e5', linewidth=1.7)
    names = [name[:-len("Classifier")] for name in names]

    for cap in summary_plot['caps']:
        shrink = 0.5
        bounds = cap.get_ydata()
        width = bounds[1] - bounds[0]
        newbounds = bounds[0] + shrink * width / 2., bounds[1] - shrink * width / 2.
        cap.set_ydata(newbounds)

    subplot.set_yticklabels(names, size=7)
    pvalue_axis.set_yticks(subplot.get_yticks())
    pvalue_axis.yaxis.grid(False)
    display_p_values = [format_p_value(pvalue) if i != bestindex else '' for (i, pvalue) in enumerate(pvalues)]
    pvalue_axis.set_yticklabels(display_p_values, size=6)
    #subplot.set_ylim([.5, len(data) + 1])
    pvalue_axis.set_ybound(subplot.get_ybound())
    if yticklabels is not None:
        yticklabels[bestindex].set_fontweight('bold')


def load_scores(dataset_path, learner_name, prefix):
    # load scores for first configuration in the file
    _scores = [-1000] * 90
    try:
        path = os.path.join(dataset_path, "data", "{0}_{1}".format(learner_name, prefix))
        print(path)
        default_file = open(path)
        scores = cPickle.load(default_file)
        _scores = scores
    except:
        pass
    return np.array(_scores)


def load_default_scores(dataset_path, learner_name):
    return load_scores(dataset_path, learner_name, 'default')


def load_optimized_scores(dataset_path, learner_name):
    return load_scores(dataset_path, learner_name, 'best')


def make_plot_improvement_over_default(dataset_path, dataset_name):
    # title = 'performance index distribution of default and optimized configurations'
    #dataset_name = 'Test_dataset'
    names = ['PassiveAggressiveClassifier', 'RadiusNeighborsClassifier', 'GaussianNBClassifier',
             'ExtraTreeEnsembleClassifier', 'SVMClassifier', 'LinearDiscriminantClassifier',
             'KNNClassifier', 'RandomForestClassifier', 'SGDClassifier',
             'LogisticRegressionClassifier', 'NearestCentroidClassifier', 'LinearSVMClassifier',
             'NuSVMClassifier', 'DecisionTreeClassifier', 'RidgeClassifier',
             'QuadraticDiscriminantClassifier', 'GradientBoostingClassifier']
    default_data = []
    optimized_data = []
    for name in names:
        default_data.append(load_default_scores(dataset_path, name))
        optimized_data.append(load_optimized_scores(dataset_path, name))

    #default_data = [np.random.normal(np.random.normal(0.45, 0.15), 0.02, np.random.randint(25, 30)) for learner in names]
    #optimized_data = [default_item + np.random.normal(0.1, 0.05, len(default_item)) for default_item in default_data]
    means = [np.mean(optimized_item) for optimized_item in optimized_data]
    pvalues = get_pvalues(optimized_data)
    #pvalues = np.random.normal(0.5, 0.0001, len(names))
    display_optimized_default_chart(dataset_name, default_data, optimized_data, means, names, pvalues, pcritical=0.85)


def get_pvalues(model_scores):
    valid_models = np.argwhere(np.array([len(model) for model in model_scores]) > 0)
    pvalues = np.zeros(len(model_scores))
    all_scores = [model for model in model_scores if len(model) > 0]
    statistics = get_statistics(all_scores)
    k = len(all_scores)
    df = max(abs(len(all_scores[0]) - k), 2)
    pvalues_valid = np.array([p_stud(q, k, df) for q in statistics])

    for i, valid_model in enumerate(valid_models):
        pvalues[valid_model] = pvalues_valid[i]

    return pvalues


def get_worst_best_optimal(dataset_path):
    learner_names = ['PassiveAggressiveClassifier', 'RadiusNeighborsClassifier',
                     'GaussianNBClassifier', 'ExtraTreeEnsembleClassifier', 'SVMClassifier',
                     'LinearDiscriminantClassifier', 'KNNClassifier', 'RandomForestClassifier',
                     'SGDClassifier', 'LogisticRegressionClassifier', 'NearestCentroidClassifier',
                     'LinearSVMClassifier', 'NuSVMClassifier', 'DecisionTreeClassifier',
                     'RidgeClassifier', 'QuadraticDiscriminantClassifier',
                     'GradientBoostingClassifier']
    default_scores = np.array([np.mean(load_default_scores(dataset_path, learner_name))
                               for learner_name in learner_names])
    optimized_scores = [np.mean(load_optimized_scores(dataset_path, learner_name))
                        for learner_name in learner_names]
    return min(default_scores[default_scores != -1000]), max(default_scores), max(optimized_scores)


def make_plot_improvement_comparison():
    #data = [[[0.5,0.8] [0.7,0.9] [0.2,0.6]] [[0.9,1] [0.2, 0.5] [0.8, 0.85]]]
    #summary_figure = plt.figure(figsize=(5.5, 6), dpi=100)
    datasets=15
    learners=17
    figure, allplots = plt.subplots(datasets, learners)
    for dataset in xrange(datasets):
        for learner in xrange(learners):
            barplot = allplots[dataset, learner].bar([[1], [2]], np.random.rand(2).T)
    #barplot.set_ylim([0,1])
    figure.subplots_adjust(hspace=0, wspace=0)

    plt.show(block=True)


def generate_comparison_table_entry(dataset_path):
    worst, best, optimal = get_worst_best_optimal(dataset_path)
    with open("figures/comparison_table.txt", 'a') as output_file:
        output_file.write("{0} & {1:.4} & {2:.4} & {3:.4} \\\\".format(dataset_name, worst, best, optimal))

if __name__ == '__main__':
    dataset_path = "/home/roger/thesis/data_analysis"
    dataset_name = "balance-scale"

    if len(sys.argv) > 1:
        dataset_path = sys.argv[-2]
        dataset_name = sys.argv[-1]

    make_plot_improvement_over_default(dataset_path, dataset_name)

    #make_plot_improvement_comparison()
    generate_comparison_table_entry(dataset_path)
