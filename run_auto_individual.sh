#!/bin/bash
##########################################################################################
#   LDB
#   
#   automation script for running RTM simulation
# 
#   args:
#   $1  NN_MODEL:   FMNIST  CIFAR   RESNET  TEST[0,1,2,5,6,35,46,135]
#   $2  LOOPS:      Number of inference iterations
#   $3  GPU:        GPU to be used (0, 1)
#
##########################################################################################

NN_MODEL="$1"                       # FMNIST    CIFAR   RESNET
## params for rtm testing
TEST_ERROR=1
TEST_RTM=1
LOOPS=$2
BLOCK_SIZE=$3

timestamp=$(date +%Y-%m-%d_%H-%M-%S)
results_dir="RTM_results/$NN_MODEL/$BLOCK_SIZE/$timestamp"
output_dir="$results_dir/outputs"
if [ ! -d "$results_dir" ]; then
    mkdir -p "$results_dir"
    if [ ! -d "$output_dir" ]; then
        mkdir -p "$output_dir"
    else
        echo "Directory $output_dir already exists."
    fi
else
    echo "Directory $results_dir already exists."
fi


if [ "$NN_MODEL" = "FMNIST" ]
then
    MODEL="VGG3"
    DATASET="FMNIST"
    declare -a PROTECT_LAYERS=(1 1 1 1)
    declare -a ERRSHIFTS=(0 0 0 0)
    MODEL_PATH="model_fmnist9108.pt"
elif [ "$NN_MODEL" = "CIFAR" ]
then
    MODEL="VGG7"
    DATASET="CIFAR10"
    declare -a PROTECT_LAYERS=(1 1 1 1 1 1 1 1) 
    declare -a ERRSHIFTS=(0 0 0 0 0 0 0 0)
    # total_blocks = (size[0]*size[1])
    # total_blocks = (384, 16384, 32768, 65536, 131072, 262144, 8388608, 10240)
    # total_elements = (size[0]*size[1]/block_size)
    # total_elements_64 = (6, 256, 512, 1024, 2048, 4096, 131072, 160)
    MODEL_PATH="model_cifar8582.pt"
    # MODEL_PATH="model_cifar8660.pt"
elif [ "$NN_MODEL" = "RESNET" ]
then 
    MODEL="ResNet"
    DATASET="IMAGENETTE"
    MODEL_PATH="model_resnet7694.pt"
elif [[ "$NN_MODEL" == *TEST* ]]; then
    MP=$(echo "$NN_MODEL" | grep -o '[0-9]\+')
    MODEL="VGG7"
    DATASET="CIFAR10"
    declare -a PROTECT_LAYERS=(1 1 1 1 1 1 1 1)
    declare -a ERRSHIFTS=(0 0 0 0 0 0 0 0)
    MODEL_PATH="model_test_cifar_mp$MP.pt"
    # echo "$MODEL_PATH"
else
    echo -e "\n\033[0;31m$NN_MODEL not supported, check spelling & available models: FMNIST, CIFAR, RESNET\033[0m\n"
    exit
fi

## default params
BATCH_SIZE=256
EPOCHS=1
LR=0.001
STEP_SIZE=25

## cont. params for rtm testing
# echo -e "${PROTECT_LAYERS[@]}"
NR_UNPROC=$4
PROTECT_LAYERS[$NR_UNPROC]=0
# echo -e "${PROTECT_LAYERS[@]}"
GPU=$5

# declare -a PERRORS=(0.0001)

declare -a PERRORS=(0.0)
# declare -a PERRORS=(0.1 0.01 0.001 0.0001)
# declare -a PERRORS=(0.001 0.0001 0.00001 0.000001)
# declare -a PERRORS=(0.0000455 0.0000995 0.000207 0.000376 0.000594 0.000843 0.0011)
# declare -a PERRORS=(0.0000455)
# declare -a PERRORS=(0.0001 0.0000455)
# declare -a PERRORS=(0.0001 0.0000455 0.00001 0.000001)


for p in "${PERRORS[@]}"
do
    echo -e "\n\033[0;32mRunning $NN_MODEL for $LOOPS loops with error: $p\033[0m\n"
    
    declare -a list
    
    for layer in "${!PROTECT_LAYERS[@]}"    
    do
        if [ "${PROTECT_LAYERS[$layer]}" == 0 ]; then
            let "L=layer+1"
            echo -e "\n\033[0;33mUNprotected Layer: $L\033[0m\n"

            # PROTECT_LAYERS[$layer]=0
            # echo "${PROTECT_LAYERS[@]}"
            
            output_dir_L="$output_dir/$L"
            if [ ! -d "$output_dir_L" ]; then
                mkdir -p "$output_dir_L"
            else
                echo "Directory $output_dir_L already exists."
            fi
            output_file="$output_dir_L/output_${DATASET}_$L-$LOOPS-$p.txt"

            python run.py --model=${MODEL} --dataset=${DATASET} --batch-size=${BATCH_SIZE} --epochs=${EPOCHS} --lr=${LR} --step-size=${STEP_SIZE} --test-error=${TEST_ERROR} --load-model-path=${MODEL_PATH} --loops=${LOOPS} --perror=$p --test_rtm=${TEST_RTM} --gpu-num=$GPU --block_size=$BLOCK_SIZE --protect_layers ${PROTECT_LAYERS[@]} --err_shifts ${ERRSHIFTS[@]} | tee "$output_file"
            
            penultimate_line=$(tail -n 2 "$output_file" | head -n 1)
            # Remove square brackets and split values
            values=$(echo "$penultimate_line" | tr -d '[]')

            list+=("$values")
            
            # echo $list
            
            python plot_new_table.py ${output_file} ${results_dir} ${NN_MODEL} ${LOOPS} ${p} ${L}

            # PROTECT_LAYERS[$layer]=1
        fi
    done

    csv_file="$output_dir/table_$p.csv"

    for value in "${list[@]}"
    do
        echo "${value[@]}" >> "$csv_file"
    done

    unset list
    
done

# echo -e "${PROTECT_LAYERS[@]}"
PROTECT_LAYERS[$NR_UNPROC]=1
# echo -e "${PROTECT_LAYERS[@]}"


# Check if an input file is provided as an argument
# if [[ $OUTPUT_FILE -ne 1 ]]; then
#     echo "Usage: $0 <input_file>"
#     exit 1
# fi

# # Check if the input file exists
# if [[ ! -f $OUTPUT_FILE ]]; then
#     echo "Input file not found: $OUTPUT_FILE"
#     exit 1
# fi

# # Read the penultimate line from the input file
# penultimate_line=$(tail -n 2 $OUTPUT_FILE | head -n 1)

# # Convert the penultimate line into an array of double values
# read -ra double_array <<< "$penultimate_line"

# # Print the double values in the array
# for value in "${double_array[@]}"; do
#     echo "$value"
# done

