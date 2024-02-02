import matplotlib.pyplot as plt
import numpy as np
import os

import sys
import ast

import re

# Get the current directory
current_directory = os.getcwd()
file = ""
data_files = []

# Iterate over all the directories in the current directory
for folder_name in os.listdir(current_directory):
    if os.path.isdir(os.path.join(current_directory, folder_name)):
        for folder_name2 in os.listdir(folder_name):
            if os.path.isdir(os.path.join(folder_name, folder_name2)):
                folder_name3=folder_name + "/" + folder_name2 + "/outputs"
                for folder_name4 in os.listdir(folder_name3):
                    if os.path.isdir(os.path.join(folder_name3, folder_name4)):
                        folder_name5=folder_name3 + "/" + folder_name4
                        for file_name in os.listdir(folder_name5):
                            file = folder_name5 + "/" + file_name
                            data_files.append(file)
# print(data_files)
# # Assuming you have 4 data files
# data_files = ['block_size64/2023-11-07_18-38-39/outputs/1/output_CIFAR10_1-1000-0.0001.txt', 
#               'block_size32/2023-11-08_12-05-22/outputs/1/output_CIFAR10_1-1000-0.0001.txt', 
#               'block_size16/2023-11-08_15-48-17/outputs/1/output_CIFAR10_1-1000-0.0001.txt', 
#               'block_size8/2023-11-09_13-59-09/outputs/1/output_CIFAR10_1-1000-0.0001.txt']

fig, axs = plt.subplots(6, 8)  # Creates a 2x2 grid of subplots


for i, ax in enumerate(axs.flat):
    # Check if the input file exists
    try:
        with open(data_files[i]) as f:
            pass
    except FileNotFoundError:
        print("Input file not found: " + data_files[i])
        sys.exit(1)

    match = re.search(r'/(\d+)/', data_files[i])
    if match:
        layer_nr = match.group(1)
    else:
        layer_nr = 0

    match = re.search(r'block_size(\d+)', data_files[i])
    if match:
        block_size = match.group(1)
    else:
        block_size = 0

    # Read the penultimate line from the input file
    with open(data_files[i]) as f:
        lines = f.readlines()
        penultimate_line = lines[-2].strip()

    # Extract the double values from the string representation of the array
    y_values = ast.literal_eval(penultimate_line)

    print(y_values)

    # loops = sys.argv[4]
    # perror = sys.argv[5]
    # unproc_layer = sys.argv[6]

    # output_dir = sys.argv[2] + "/figures/" + unproc_layer + "/"
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    # nn_model = sys.argv[3]
    nn_model = "CIFAR"
    if(nn_model == "FMNIST"):
        baseline = 91.08 
    elif (nn_model == "CIFAR"):
        baseline = 85.85
    elif (nn_model == "RESNET"):
        baseline = 76.94
    else:
        baseline = 0.0

    # filename = "graph_" + str(nn_model) + "-" + str(unproc_layer) + "-" + str(loops) + "-" + str(perror)

    x_values = range(len(y_values))

    ax.plot(x_values, y_values)
    ax.set_yticks(range(0, 100, 5))
    arg3 = int(len(x_values)/10)
    if arg3 == 0:
        arg3 = 1
    ax.set_xticks(range(0, len(x_values) + 1, arg3))
    ax.axhline(y=baseline, color='r', linestyle='--')

    ax.grid(True)
    ax.set_xlabel('Iteration #')
    ax.set_ylabel('Accuracy')
    ax.set_title("block_size" + str(block_size) + " | layer " + str(layer_nr))

fig.set_size_inches(60,40)
plt.savefig("subgraphs.png")

# plt.show()