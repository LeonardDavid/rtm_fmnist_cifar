import matplotlib.pyplot as plt
import os

import sys
import ast

# Check if an input file is provided as an argument
# if len(sys.argv) != 2:
#     print("Usage: " + sys.argv[0] + " <input_file>")
#     sys.exit(1)

# Check if the input file exists
try:
    with open(sys.argv[1]) as f:
        pass
except FileNotFoundError:
    print("Input file not found: " + sys.argv[1])
    sys.exit(1)

# Read the penultimate line from the input file
with open(sys.argv[1]) as f:
    lines = f.readlines()
    penultimate_line = lines[-2].strip()

# Extract the double values from the string representation of the array
y_values = ast.literal_eval(penultimate_line)

print(y_values)

# # Print the double values in the array
# for value in y_values:
#     print(value)

output_dir = sys.argv[2] + "/figures/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

nn_model = sys.argv[3]
if(nn_model == "FMNIST"):
    baseline = 91.08 # 9108/8582/7694
elif (nn_model == "CIFAR"):
    baseline = 85.85
elif (nn_model == "RESNET"):
    baseline = 76.94
else:
    baseline = 0.0

loops = sys.argv[4]
perror = sys.argv[5]

filename = "graph_" + str(nn_model) + "_" + str(loops) + "_" + str(perror)

## vgg3 fmnist fc1 fc2
# y_values = [85.76, 85.5, 85.57000000000001, 85.47, 85.38, 85.36, 85.05, 85.26, 85.07000000000001, 84.97, 84.68, 85.11999999999999, 84.96000000000001, 84.84, 84.89999999999999, 84.98, 85.26, 85.07000000000001, 84.89999999999999, 85.11, 85.08, 84.94, 84.83000000000001, 84.75, 84.74000000000001, 84.56, 84.45, 84.22, 83.91, 83.74000000000001, 83.89999999999999, 83.77, 83.62, 83.61, 83.25, 83.42, 83.46000000000001, 82.96, 82.37, 81.67999999999999, 80.67999999999999, 80.21000000000001, 80.55, 80.46, 80.95, 81.13, 80.73, 80.35, 79.42, 79.27]
# y_values = [85.91, 85.91, 85.85000000000001, 85.76, 85.82, 85.92, 85.54, 85.65, 85.79, 85.76, 85.76, 85.61999999999999, 85.47, 85.47, 85.54, 85.61999999999999, 85.42, 85.19, 85.26, 85.11999999999999, 84.85000000000001, 84.78999999999999, 84.56, 84.28999999999999, 84.2, 84.61999999999999, 84.61, 84.34, 84.31, 84.2, 83.78999999999999, 83.88, 83.89, 83.7, 83.47, 83.47, 83.08, 82.65, 82.39999999999999, 82.39999999999999, 82.45, 82.24000000000001, 82.05, 81.65, 81.38, 81.21000000000001, 80.25999999999999, 79.72, 79.17999999999999, 78.79, 78.88, 78.96, 79.29, 79.05, 78.86, 78.78, 78.42, 78.25999999999999, 78.23, 77.49000000000001, 76.4, 75.75, 75.0, 74.81, 74.55000000000001, 74.15, 73.56, 73.15, 72.82, 72.81, 72.08, 70.97, 71.27, 71.41999999999999, 72.11, 72.41, 72.17, 72.03, 71.87, 71.58, 71.02000000000001, 70.09, 70.22, 69.73, 69.6, 69.44, 69.46, 68.77, 68.58999999999999, 69.15, 68.33, 67.17, 66.95, 66.47999999999999, 66.18, 65.73, 65.83, 65.58, 65.44, 64.46]
# y_values = [83.14, 70.03, 59.78, 47.08, 34.050000000000004, 28.18, 23.34, 24.75, 18.32, 17.080000000000002, 14.29, 17.5, 23.25, 22.7, 21.47, 16.75, 20.25, 20.06, 21.21, 20.39, 18.04, 14.81, 10.6, 15.17, 16.66, 18.45, 16.520000000000003, 19.07, 16.220000000000002, 16.57, 9.51, 10.38, 12.24, 16.17, 18.37, 11.89, 10.27, 10.59, 11.14, 11.600000000000001, 13.36, 11.03, 12.46, 16.259999999999998, 24.81, 24.62, 20.79, 22.650000000000002, 25.45, 18.63, 11.64, 14.67, 14.71, 14.21, 9.59, 13.79, 14.99, 11.25, 12.559999999999999, 12.18, 10.59, 10.36, 12.85, 12.86, 14.14, 13.100000000000001, 13.320000000000002, 14.360000000000001, 14.93, 14.01, 13.69, 10.63, 13.76, 16.37, 15.42, 11.61, 11.219999999999999, 13.94, 9.969999999999999, 13.48, 11.87, 12.61, 15.260000000000002, 12.29, 12.709999999999999, 13.639999999999999, 14.59, 14.96, 16.509999999999998, 14.299999999999999, 10.57, 12.17, 12.55, 11.459999999999999, 10.82, 13.65, 13.930000000000001, 13.489999999999998, 12.23, 11.16]
# y_values = [85.77, 85.75, 85.76, 85.8, 85.81, 85.82, 85.65, 85.85000000000001, 85.79, 85.78, 85.72999999999999, 85.54, 85.58, 85.63, 85.61999999999999, 85.64, 85.67, 85.61, 85.58, 85.57000000000001, 85.6, 85.56, 85.52, 85.52, 85.53, 85.50999999999999, 85.52, 85.52, 85.47, 85.48, 85.42999999999999, 85.46000000000001, 85.42, 85.35000000000001, 85.28999999999999, 85.27, 85.2, 85.18, 85.21, 85.19, 85.11, 85.08, 85.13, 85.1, 85.08, 85.08, 85.13, 85.14, 85.15, 85.22999999999999, 85.11, 85.11999999999999, 85.11999999999999, 84.95, 84.72, 84.78999999999999, 84.72, 84.77, 84.72, 84.69, 84.71, 84.68, 84.59, 84.58, 84.68, 84.73, 84.61999999999999, 84.6, 84.66, 84.67, 84.64, 84.7, 84.52, 84.39999999999999, 84.37, 84.39, 84.22, 84.32, 84.38, 84.39, 84.26, 84.50999999999999, 84.54, 84.52, 84.54, 84.5, 84.53, 84.48, 84.45, 84.38, 84.49, 84.56, 84.59, 84.71, 84.78999999999999, 84.84, 84.75, 84.72, 84.67, 84.72]

## vgg3 fmnist conv1 conv2 fc1 fc2 50 0.001
# y_values = [85.48, 85.32, 84.97, 84.67, 84.53, 83.59, 83.09, 82.86, 81.73, 82.28, 81.39, 79.97999999999999, 78.0, 74.58, 72.11, 72.28, 71.43, 70.27, 68.44, 65.59, 64.8, 64.25999999999999, 63.9, 63.970000000000006, 63.36000000000001, 64.25999999999999, 64.23, 63.14999999999999, 61.870000000000005, 60.589999999999996, 59.93000000000001, 59.64, 61.53999999999999, 60.86, 58.53, 53.56999999999999, 49.68, 48.449999999999996, 50.32, 50.67, 50.73, 52.33, 49.0, 46.64, 47.620000000000005, 50.93, 50.519999999999996, 50.93, 51.519999999999996, 50.79]
## vgg3 fmnist conv1 conv2 fc1 fc2 50 0.0001
# y_values = [85.76, 85.71, 85.7, 85.7, 85.50999999999999, 85.57000000000001, 85.59, 85.71, 85.71, 85.72, 85.8, 85.72999999999999, 85.69, 85.74000000000001, 85.69, 85.78, 85.65, 85.64, 85.65, 85.61, 85.72, 85.78, 85.69, 85.67, 85.66, 85.58, 85.26, 85.17, 85.16, 85.15, 85.09, 85.05, 85.02, 84.73, 84.68, 84.74000000000001, 84.71, 83.81, 83.86, 83.91, 83.84, 83.83, 83.89999999999999, 83.89, 83.91999999999999, 83.97, 83.97, 84.07, 84.05, 84.14]

x_values = range(len(y_values))


plt.plot(x_values, y_values)
plt.yticks(range(0, 100, 5))
# plt.yticks(range(80, 90, 1))
plt.xticks(range(0, len(x_values) + 1, int(len(x_values)/10)))
plt.axhline(y=baseline, color='r', linestyle='--')

plt.grid(True)
plt.xlabel('Iteration #')
plt.ylabel('Accuracy')
plt.title('Accumulated shift for ' + str(nn_model) + ' over ' + str(loops) + ' inference iterations with error ' + str(perror))

fig = plt.gcf()
fig.set_size_inches(15,9)
plt.savefig(output_dir + filename + ".png")

# plt.show()