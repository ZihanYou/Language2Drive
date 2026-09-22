export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=/path/to/your/project/root:$PYTHONPATH

echo "Start generating dataset..."

/path/to/your/python/bin/python \
    data_tools/senna_nusc_data_converter.py \
    nuscenes \
    --root-path /path/to/nuscenes \
    --out-dir /path/to/your/output \
    --extra-tag senna_nusc \
    --version v1.0 \
    --canbus /path/to/nuscenes/canbus
