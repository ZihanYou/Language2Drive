export PYTHONPATH=/path/to/Senna:$PYTHONPATH
export PATH=/path/to/your/python/bin:$PATH

CUDA_VISIBLE_DEVICES=1 python eval_tools/senna_plan_cmd_eval_multi_img.py
