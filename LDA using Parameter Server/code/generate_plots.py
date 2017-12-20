'''
Generates all the rewuired plots
'''

import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt

data_dir = sys.argv[1]
output_dir = sys.argv[2]

subdirs = glob.glob(data_dir + '/*/')
print subdirs

for subdir in subdirs:
    # print "subdir = ", subdir
    # cur_dir = os.path.join(data_dir, subdir)
    # print "cur_dir = ", cur_dir
    file_path = os.path.join(subdir, 'likelihood_0-0.csv')
    # print "file_path = ", file_path
    output_file_base = subdir.split('/')[1] + '_plot_'

    data = pd.read_csv(file_path, header=None)
    iterations = list(data[0])
    time = list(data[1])
    llh = list(data[2])

    # Plot epoch vs likelihood
    plt.close()
    plt.plot(iterations, llh)
    plt.ylabel('Log Likelihood')
    plt.xlabel('Iterations')
    plot_file = os.path.join(output_dir, output_file_base + 'llh_vs_epoch.png')
    plt.savefig(plot_file)

    # Plot Likelihood vs Time
    plt.close()
    plt.plot(time, llh)
    plt.ylabel('Log Likelihood')
    plt.xlabel('Time Elapsed')
    plot_file = os.path.join(output_dir, output_file_base + 'llh_vs_time.png')
    plt.savefig(plot_file)




