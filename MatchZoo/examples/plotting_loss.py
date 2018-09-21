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

def draw_train_learning_curve(log_file):
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 14
    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    marker = itertools.cycle(('b-+', 'g-^', 'r-v', 'm-8', '-*', '-.', 'r-,', 'g-v', 'b-*', 'c-v', 'b-o', 'c-s', '-v'))
    handles_lines = []

    for f in os.listdir(log_file):
        info = open(os.path.join(log_file, f), 'r')
        start_line = '[Model] Model Compile Done.'
        start_flag = False
        loss = []
        for line in info:
            line = line.strip()
            # print(line)
            if start_flag:
                # print (line)
                tokens = line.split('\t')
                if len(tokens) > 1:
                    # print('tokens: ', tokens)
                    if 'train' in line and 'loss' in line:
                        # print(tokens)
                        loss.append(float(tokens[2].split('=')[1]))
            if start_line in line.strip():
                start_flag = True
                # print(line)

        # draw learning curve
        loss = loss[:250]
        x = range(len(loss))
        # print ('x', x)
        # print (train["loss"])
        line, = plt.plot(x, loss, marker.__next__(), label=f.split('.')[0])
        handles_lines.append(line)

    plt.legend(handles=handles_lines, loc=1)
    rcParams['grid.linestyle'] = 'dotted'
    plt.grid()

    min_loss = min(loss)
    plt.ylabel('Model training loss curves')
    # log_label = "Performance evolution of " + log_file.split('/')[-1].split('.')[0]
    plt.title(log_label)
    plt.xlabel('Training epochs')
    plt.show()
    # plt.savefig(log_file.split('.')[0]+".png")


exp_id = 1
print('Exp ', exp_id, ': model comparation')
log_file = sys.argv[1]  # file.log containing training details
log_label = sys.argv[2]
draw_train_learning_curve(log_file)

print("Finished.")
