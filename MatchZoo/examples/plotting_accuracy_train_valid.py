"""
Parse the log files of MatchZoo to draw the learning curves of
deep learning model training process to observe the training loss
and metrics on train/dev/test data. With these curves, we can do
a better job in model debugging
@author: Liu Yang (yangliuyx@gmail.com / lyang@cs.umass.edu)
@author: Thiziri Belkacem (belkacemthiziri@gmail.com)
"""

import os
import sys
import matplotlib.pyplot as plt
from matplotlib import rcParams
import itertools

def draw_train_learning_curve(log_file, measures):
    info = open(log_file, 'r')
    start_line = '[Model] Model Compile Done.'
    start_flag = False
    train = {measure : [] for measure in measures}
    train["loss"] = []
    valid = {measure: [] for measure in measures}
    test = {measure: [] for measure in measures}
    for line in info:
        line = line.strip()
        # print(line)
        if start_flag:
            # print (line)
            tokens = line.split('\t')
            if len(tokens) > 1:
                # print('tokens: ', tokens)
                if 'train' in line:
                    # print(tokens)
                    train["loss"].append(float(tokens[2].split('=')[1]))
                elif 'valid' in line:
                    # print (tokens)
                    if len(tokens) < len(measures)+2:
                        continue
                    valid_token = {}
                    for measure in measures:
                        valid_token[measure] = [token for token in tokens if measure in token]
                    for measure in measures:
                        valid[measure].append(float(valid_token[measure][0].split('=')[1]))
                elif 'test' in line:
                    # print (tokens)
                    if len(tokens) < len(measures) + 2:
                        continue
                    test_token = {}
                    for measure in measures:
                        test_token[measure] = [token for token in tokens if measure in token]
                    for measure in measures:
                        test[measure].append(float(test_token[measure][0].split('=')[1]))
        if start_line in line.strip():
            start_flag = True
            # print(line)

    # draw learning curve
    SMALL_SIZE = 8
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    x = range(len(train["loss"]))
    # print ('x', x)
    # print (train["loss"])
    if (len(train["loss"]) != len(test[measures[0]])) or (len(train["loss"]) != len(valid[measures[0]])):
        print('wait a few seconds for the valid/test metrics in the next iteration...')
        min_len = min(len(test[measures[0]]), len(valid[measures[0]]))
        train["loss"] = train["loss"][0:min_len]
        for measure in valid:
            valid[measure] = valid[measure][0:min_len]
        for measure in test:
            test[measure] = test[measure][0:min_len]
        x = x[0:min_len]
    print(len(train["loss"]), [len(test[measure]) for measure in test], [len(valid[measure]) for measure in valid])
    handles_lines = []
    marker = itertools.cycle(('b-+', 'g-^', 'r-v', 'm-8', '-*', '-.', 'r-,', 'g-v', 'b-*', 'c-v', 'b-o', 'c-s', '-v'))
    # line, = plt.plot(x, train["loss"], marker.__next__(), label='train["loss"]')
    # handles_lines.append(line)
    for measure in valid:
        line, = plt.plot(x, valid[measure], marker.__next__(), label='valid_'+measure)
        handles_lines.append(line)
    for measure in test:
        line, = plt.plot(x, test[measure], marker.__next__(), label='test_'+measure)
        handles_lines.append(line)

    plt.hlines(y=0.76, colors='red', xmin=0, xmax=500)
    plt.legend(handles=handles_lines, loc=(0.7,0.5))
    rcParams['grid.linestyle'] = 'dotted'
    plt.grid()

    plt.ylabel('Accuracy values')

    log_label = " " + log_file.split('/')[-1].split('.')[0]
    plt.title(log_label)
    plt.xlabel('Training epochs')
    plt.show()
    # plt.savefig(log_file.split('.')[0]+".png")


exp_id = 1
print('Exp ', exp_id, ': model comparation')
measures = sys.argv[2].split(',')  # must be list of measures separated with ',' ex: map,p@5
if not os.path.isfile(sys.argv[1]):
    for model in os.listdir(sys.argv[1]):
        if '.log' in model:
            log_file = os.path.join(sys.argv[1], model)  # 'path_of_log_file'
            log_label = model + ' training'
            draw_train_learning_curve(log_file, measures)
else:
    log_file = sys.argv[1]  # file.log containing training details
    draw_train_learning_curve(log_file, measures)

print("Finished.")
