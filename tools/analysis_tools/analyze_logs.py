# Copyright (c) OpenMMLab. All rights reserved.
"""Modified from https://github.com/open-
mmlab/mmdetection/blob/master/tools/analysis_tools/analyze_logs.py."""
import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns


def plot_curve(log_dicts, args):
    if args.backend is not None:
        plt.switch_backend(args.backend)
    # sns.set_style(args.style)
    # if legend is None, use {filename}_{key} as legend
    legend = args.legend
    if legend is None:
        legend = []
        for json_log in args.json_logs:
            for metric in args.keys:
                legend.append(f'{json_log}_{metric}')
    assert len(legend) == (len(args.json_logs) * len(args.keys))
    metrics = args.keys

    num_metrics = len(metrics)
    for i, log_dict in enumerate(log_dicts):
        epochs = list(log_dict.keys())
        for j, metric in enumerate(metrics):
            print(f'plot curve of {args.json_logs[i]}, metric is {metric}')
            plot_epochs = []
            plot_iters = []
            plot_values = []
            # In some log files exist lines of validation,
            # `mode` list is used to only collect iter number
            # of training line.
            for epoch in epochs:
                epoch_logs = log_dict[epoch]
                if metric not in epoch_logs.keys():
                    continue
                if metric in ['mIoU', 'mAcc', 'aAcc']:
                    plot_epochs.append(epoch)
                    plot_values.append(epoch_logs[metric][0])
                else:
                    for idx in range(len(epoch_logs[metric])):
                        plot_iters.append(epoch_logs['step'][idx])
                        plot_values.append(epoch_logs[metric][idx])
            ax = plt.gca()
            label = legend[i * num_metrics + j]

            if metric in ['mIoU', 'mAcc', 'aAcc']:
                ax.set_xticks(plot_epochs)
                plt.xlabel('step', fontsize=25)
                plt.ylabel('mIoU(%)', fontsize=25)
                plt.plot(plot_epochs, plot_values, label=label.split('.')[0], marker='*')
                x_labels = ["20k", "40k", "60k", "80k"]
                x_values = [20000, 40000, 60000, 80000]
                plt.xticks(x_values, x_labels, fontsize=25)
                y_labels = ["20k", "40k", "60k", "80k"]
                y_values = [78, 83, 88, 93]
                # plt.yticks(y_values, y_labels, fontsize=20)
                plt.yticks(y_values, fontsize=25)
            else:
                x_labels = ["20k", "40k", "60k", "80k"]
                x_values = [20000, 40000, 60000, 80000]
                plt.xticks(x_values, x_labels, fontsize=14)
                plt.yticks(fontsize=20)
                plt.xlabel('iter', fontsize=20)  # 设置 x 轴标签字体大小
                plt.ylabel('acc', fontsize=20)  # 设置 y 轴标签字体大小
                plt.title('acc', fontsize=20)
                plt.plot(plot_iters, plot_values, label=label, linewidth=0.5)
        plt.legend(fontsize=13, frameon=False)
        plt.rcParams['figure.facecolor'] = 'white'
        if args.title is not None:
            plt.title(args.title)
    if args.out is None:
        plt.show()
    else:
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        print(f'save curve to: {args.out}')
        plt.savefig(args.out)
        plt.cla()


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Json Log')
    parser.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format')
    parser.add_argument(
        '--keys',
        type=str,
        nargs='+',
        default=['mIoU'],
        help='the metric that you want to plot')
    parser.add_argument('--title', type=str, help='title of figure')
    parser.add_argument(
        '--legend',
        type=str,
        nargs='+',
        default=None,
        help='legend of each plot')
    parser.add_argument(
        '--backend', type=str, default=None, help='backend of plt')
    parser.add_argument(
        '--style', type=str, default='dark', help='style of plt')
    parser.add_argument('--out', type=str, default='mIoU.png')
    args = parser.parse_args()
    return args


def load_json_logs(json_logs):
    # load and convert json_logs to log_dict, key is step, value is a sub dict
    # keys of sub dict is different metrics
    # value of sub dict is a list of corresponding values of all iterations
    log_dicts = [dict() for _ in json_logs]
    prev_step = 0
    for json_log, log_dict in zip(json_logs, log_dicts):
        with open(json_log) as log_file:
            for line in log_file:
                log = json.loads(line.strip())
                # the final step in json file is 0.
                if 'step' in log and log['step'] != 0:
                    step = log['step']
                    prev_step = step
                else:
                    step = prev_step
                if step not in log_dict:
                    log_dict[step] = defaultdict(list)
                for k, v in log.items():
                    log_dict[step][k].append(v)
    return log_dicts


def main():
    args = parse_args()
    json_logs = args.json_logs
    for json_log in json_logs:
        assert json_log.endswith('.json')
    log_dicts = load_json_logs(json_logs)
    # plt.figure()
    plt.rcParams.update({'font.size': 15})
    plot_curve(log_dicts, args)



if __name__ == '__main__':
    main()
