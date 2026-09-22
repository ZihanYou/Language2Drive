import os
from moviepy import VideoFileClip
from pathlib import Path


def convert_mp4_to_gif(input_path, output_path, fps=10):
    """
    将MP4视频转换为GIF
    :param input_path: 输入MP4文件路径
    :param output_path: 输出GIF文件路径
    :param fps: GIF的帧率
    """
    try:
        # 加载视频
        video = VideoFileClip(input_path)

        # 转换为GIF
        video.write_gif(output_path, fps=fps)

        # 关闭视频文件
        video.close()

        print(f"成功转换: {input_path} -> {output_path}")
    except Exception as e:
        print(f"转换失败 {input_path}: {str(e)}")


def process_directory(input_dir, output_dir):
    """
    处理目录下的所有MP4文件
    :param input_dir: 输入目录
    :param output_dir: 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有MP4文件
    mp4_files = list(Path(input_dir).glob("**/*.mp4"))

    for mp4_file in mp4_files:
        # 构建输出路径
        relative_path = mp4_file.relative_to(input_dir)
        output_path = Path(output_dir) / relative_path.with_suffix(".gif")

        # 确保输出文件的父目录存在
        os.makedirs(output_path.parent, exist_ok=True)

        # 转换文件
        convert_mp4_to_gif(str(mp4_file), str(output_path))


if __name__ == "__main__":
    # 设置输入和输出目录
    input_directory = "videos"
    output_directory = "assets/gifs"

    # 处理所有视频
    process_directory(input_directory, output_directory)
