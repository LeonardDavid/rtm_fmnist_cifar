#!/bin/bash

# Specify the number of total layers in NN model
layers=8        ###
# Specify block_size for the RTM nanowire
block_size=2   ###

# Loop through all layers individually
for ((layer=0; layer<layers; layer++))
do
    # Run the bash file
    #   params:
    ##  $1: NN model name
    ##  $2: loops
    ##  $3: block_size
    ##  $4: unprotected layer
    ##  $5: GPU used
    bash run_auto_individual.sh CIFAR 100 $block_size $layer 0
done