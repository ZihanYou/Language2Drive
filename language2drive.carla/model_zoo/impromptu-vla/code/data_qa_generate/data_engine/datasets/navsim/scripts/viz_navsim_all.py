# from data_engine.datasets.navsim.dataset_navsim import VLMNavsim, v0_container_out_key_comb, v0_pipelines

# if __name__ == '__main__':
#     import os
#     import json
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--jsonl_file", type=str, required=True)
#     parser.add_argument("--output_file", type=str, required=True)
#     parser.add_argument("--viz_every", type=int, default=100)   #! not implemented yet
#     # parser.add_argument("--viz_dir", type=str, default="viz")
#     args = parser.parse_args()
    
#     dataset_test = VLMNavsim(mode="test", pipelines=v0_pipelines, container_out_key_comb=v0_container_out_key_comb)
    
#     results = dataset_test.evaluate(jsonl_file=args.jsonl_file)
    
#     os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
#     with open(args.output_file, "w") as f:
#         f.write(json.dumps(results, indent=4, ensure_ascii=False))
from data_engine.datasets.navsim.dataset_navsim import VLMNavsim, v0_container_out_key_comb, v0_pipelines



if __name__ == '__main__':
    import os
    import json
    import argparse
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, message=".*To copy construct from a tensor.*")


    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--viz_every", type=int, default=100)
    args = parser.parse_args()

    # 构造数据集 + pipeline
    dataset_test = VLMNavsim(
        mode="test",
        pipelines=v0_pipelines,
        container_out_key_comb=v0_container_out_key_comb,
    )

    # 设置可视化路径（将 output_file 的 .json 替换成文件夹）
    viz_dir = os.path.join(
        os.path.dirname(args.output_file),
        os.path.basename(args.output_file).replace(".json", "")
    )
    if args.viz_every > 0:
        os.makedirs(viz_dir, exist_ok=True)

    # 把可视化频率和目录传递给对应 pipeline（假设最后一个 pipeline 是 planning）
    dataset_test.pipelines[-1].viz_every_eval = args.viz_every
    dataset_test.pipelines[-1].viz_path = viz_dir

    # 执行评估
    results = dataset_test.viz_all_results(jsonl_file=args.jsonl_file, interval=args.viz_every)

    # # 保存结果
    # os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    # with open(args.output_file, "w") as f:
    #     f.write(json.dumps(results, indent=4, ensure_ascii=False))

    # # 打包可视化结果
    # if args.viz_every > 0:
    #     os.system(f"zip -qr {viz_dir}.zip {viz_dir}")
