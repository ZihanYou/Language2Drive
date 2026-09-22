export LD_LIBRARY_PATH=/path/to/your/python/lib/python3.8/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0 

/path/to/your/python/bin/python eval_tools/senna_plan_visualization.py \
    --eval-data-path /path/to/your/eval/data/eval_qa.json \
    --model-path /path/to/your/model/senna-llava-v1.5-7b
    --save-path /path/to/save/images