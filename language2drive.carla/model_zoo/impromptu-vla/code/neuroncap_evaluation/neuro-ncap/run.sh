#################################################################
# Edit the following paths to match your setup
qwen_ckpt_path=$1
RUNS=${2:-10} 
CUSTOM_SUFFIX=${3:-""} 
BASE_DIR='/path/to/your/NeuroNCAP'
NUSCENES_PATH='/path/to/your/NeuroNCAP/neuro-ncap/data/nuscenes'
# Model related stuff
MODEL_NAME='MODEL_NAME'
MODEL_FOLDER=$BASE_DIR/$MODEL_NAME

# Rendering related stuff
RENDERING_FOLDER=$BASE_DIR/'neurad-studio'
RENDERING_CHECKPOITNS_PATH=$BASE_DIR/'neurad-studio/checkpoints'

# NCAP related stuff
NCAP_FOLDER=$BASE_DIR/'neuro-ncap'

# server port file
PORT_FILE='/path/to/your/port.txt'

#################################################################

# SLURM related stuff
CLEAN_PATH=$(echo "$qwen_ckpt_path" | sed 's|/$||')
LAST_DIR=$(basename "$CLEAN_PATH")
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
if [ -n "$CUSTOM_NAME" ]; then
    NAME="$CUSTOM_NAME"
else
    NAME="${LAST_DIR}_${CUSTOM_SUFFIX}_${TIMESTAMP}"
fi
PORT_FILE="workdir/${NAME}/port.txt"
PAST_POS_PATH="workdir/${NAME}/past_pos.npy"
EGO_STATUS_PATH="workdir/${NAME}/ego_status.txt"

if [ -z "$qwen_ckpt_path" ]; then
    echo "Usage: $0 <qwen_ckpt_path> [runs] [custom_suffix] [custom_name] [ablation_gray]"
    echo "  qwen_ckpt_path: Path to Qwen checkpoint"
    echo "  runs: Number of runs (default: 1)"
    echo "  custom_suffix: Custom suffix for NAME (default: empty)"
    echo "  custom_name: Custom NAME to use (default: auto-generated)"
    echo "  ablation_gray: Enable ablation study with solid gray images (default: empty)"
    exit 1
fi

echo "Using RUNS=$RUNS"
echo "Using CUSTOM_SUFFIX='$CUSTOM_SUFFIX'"
echo "Generated NAME=$NAME"

# assert we are standing in the right folder, which is NCAP folder
if [ $PWD != $NCAP_FOLDER ]; then
    echo "Please run this script from the NCAP folder"
    exit 1
fi

# assert all the other folders are present
if [ ! -d $MODEL_FOLDER ]; then
    echo "Model folder not found"
    exit 1
fi

if [ ! -d $RENDERING_FOLDER ]; then
    echo "Rendering folder not found"
    exit 1
fi

/path/to/your/envs/sglang/bin/python /path/to/your/NeuroNCAP/Impromptu/inference/launch_server.py --qwen_ckpt_path $qwen_ckpt_path --port_file $PORT_FILE

qwen_infer_port=$(cat $PORT_FILE)
echo "--------------qwen_infer_port is $qwen_infer_port---------------------"

for SCENARIO in "stationary" "frontal" "side"; do
    array_file=ncap_slurm_array_$SCENARIO
    id_to_seq=scripts/arrays/${array_file}.txt
#stationary_num = 10; frontal = 5; side_num = 5
    if [ $SCENARIO == "stationary" ]; then
        num_scenarios=10
    elif [ $SCENARIO == "frontal" ]; then
        num_scenarios=5
    elif [ $SCENARIO == "side" ]; then
        num_scenarios=5
    fi
    for i in $(seq 1 $num_scenarios); do
        sequence=$(awk -v ArrayTaskID=$i '$1==ArrayTaskID {print $2}' $id_to_seq)
        if [ -z $sequence ]; then
            echo "undefined sequence"
            exit 0
        fi
        output_dir="output/$NAME/$SCENARIO-$sequence"
        completed_runs=0
        if [ -d "$output_dir" ]; then
            for run_num in $(seq 0 $((RUNS-1))); do
                if [ -d "$output_dir/run_$run_num" ]; then
                    completed_runs=$((completed_runs + 1))
                fi
            done
        fi
        
        if [ $completed_runs -eq $RUNS ]; then
            echo "Skipping scenario $SCENARIO with sequence $sequence - all $RUNS runs already completed"
            continue
        fi
        if [ $completed_runs -gt 0 ] && [ $completed_runs -lt $RUNS ]; then
            echo "Found incomplete runs in: $output_dir"
            echo "Current runs: $completed_runs/$RUNS"
            echo "Do you want to clear all existing run_* directories in this path? (Y/N)"
            read -r response
            if [[ "$response" =~ ^[Yy]$ ]]; then
                echo "Clearing existing run directories in $output_dir"
                rm -rf "$output_dir"/run_*
                completed_runs=0
            fi
        fi
        echo "Running scenario $SCENARIO with sequence $sequence (completed: $completed_runs/$RUNS)"
        BASE_DIR=$BASE_DIR\
         NUSCENES_PATH=$NUSCENES_PATH\
         MODEL_NAME=$MODEL_NAME\
         MODEL_FOLDER=$MODEL_FOLDER\
         RENDERING_FOLDER=$RENDERING_FOLDER\
         RENDERING_CHECKPOITNS_PATH=$RENDERING_CHECKPOITNS_PATH\
         NCAP_FOLDER=$NCAP_FOLDER\
         NAME=$NAME\
         qwen_infer_port=$qwen_infer_port\
         PAST_POS_PATH=$PAST_POS_PATH\
         EGO_STATUS_PATH=$EGO_STATUS_PATH\
         qwen_ckpt_path=$qwen_ckpt_path\
         ABLATION_GRAY=$ABLATION_GRAY\
         scripts/run_local_render.sh $sequence $SCENARIO --scenario-category=$SCENARIO --runs $RUNS
        #exit 0
    done
done
kill $MODEL_PID