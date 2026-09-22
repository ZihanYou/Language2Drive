from data_engine.datasets.nuscenes.dataset_nuscenes import VLMNuScenes

if __name__ == '__main__':
    import os
    import json
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--mode", type=str, default="dist-dtheta", choices=["dist-dtheta", "dist-theta", "polar", "dx-dy", "dist-curvature", "x-y"])
    parser.add_argument("--use_image", type=str, default="6v", choices=["6v", "3v", "t-6v", "t-1v"])
    parser.add_argument("--viz_every", type=int, default=100)
    # parser.add_argument("--viz_dir", type=str, default="viz")
    args = parser.parse_args()
    
    pipelines = [
        {"type": "metadata", "use_image": args.use_image},
        {"type": "ego_status", "mode": args.mode},
        {"type": "planning", "mode": args.mode},
    ]

    v0_container_out_key_comb = ["planning"]
    
    dataset_test = VLMNuScenes(mode="test", pipelines=pipelines, container_out_key_comb=v0_container_out_key_comb,)
    
    dataset_test.pipelines[2].viz_every_eval = args.viz_every
    # viz_dir: args.output_file but replace .json with a directory
    viz_dir = os.path.join(os.path.dirname(args.output_file), os.path.basename(args.output_file).replace(".json", ""))
    dataset_test.pipelines[2].viz_path = viz_dir
    if args.viz_every > 0:
        os.makedirs(viz_dir, exist_ok=True)
    
    assert os.path.exists(args.jsonl_file), f"File {args.jsonl_file} not found"
   
    results = dataset_test.evaluate(jsonl_file=args.jsonl_file)
    # dump results to output_file
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        f.write(json.dumps(results, indent=4, ensure_ascii=False))

    # zip {os.path.dirname(args.output_file)} to {os.path.dirname(args.output_file)}.zip
    os.system(f"zip -qr {viz_dir}.zip {viz_dir}")
    