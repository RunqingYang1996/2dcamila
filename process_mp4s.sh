#!/usr/bin/env bash
# 需求：
# - 对 1.mp4 ~ 50.mp4 每个建一个同名目录（如 1/、2/ ...）
# - 目录内创建子目录：images、wplus、audiochunks、audiofeatures、wplus_smooth
# - images：按 25fps 将对应 mp4 抽帧为 000001.jpg、000002.jpg ...
# - audiochunks：按 1 秒切割音频到 000001.mp3、000002.mp3 ...（最后一段可 <1s）
# - 其余目录保持为空

set -u  # 未定义变量报错
# 注：不使用 `set -e`，以便某个文件缺失时继续处理其它文件

# 检查 ffmpeg/ffprobe
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "错误：未找到 ffmpeg，请先安装（如：sudo apt-get update && sudo apt-get install -y ffmpeg）"
  exit 1
fi

# 循环处理 1.mp4 ~ 50.mp4
for i in $(seq 60 69); do
  mp4="${i}.mp4"
  if [[ ! -f "$mp4" ]]; then
    echo "跳过：未找到文件 ${mp4}"
    continue
  fi

  workdir="${i}"
  images_dir="${workdir}/images"
  wplus_dir="${workdir}/wplus"
  audiochunks_dir="${workdir}/audiochunks"
  audiofeatures_dir="${workdir}/audiofeatures"
  wplus_smooth_dir="${workdir}/wplus_smooth"

  # 创建目录
  mkdir -p "$images_dir" "$wplus_dir" "$audiochunks_dir" "$audiofeatures_dir" "$wplus_smooth_dir"

  echo "处理 ${mp4} -> 目录 ${workdir}/"

  # 1) 抽帧为 JPG（25fps，序号从 000001 开始）
  # -vf fps=25 ：重采样到 25fps
  # -q:v 2     ：高质量 JPG（数值越小质量越好）
  ffmpeg -y -hide_banner -loglevel error \
    -i "$mp4" \
    -vf "fps=25" \
    -start_number 1 \
    -q:v 2 \
    "${images_dir}/%06d.jpg" \
    && echo "  抽帧完成 -> ${images_dir}/%06d.jpg" \
    || echo "  抽帧失败（但继续后续任务）"

  # 2) 音频按 1s 切片为 MP3（000001.mp3 起始；最后一段可 <1s）
  # 说明：
  # -map a:0      ：只取第一路音频（若无音频会失败，但不影响其它视频处理）
  # -vn           ：不处理视频流
  # -c:a libmp3lame -b:a 192k ：编码为 mp3（你可改码率）
  # -f segment -segment_time 1 ：按 1s 分段
  # -segment_start_number 1    ：编号从 1 开始 -> 000001.mp3
  ffmpeg -y -hide_banner -loglevel error \
    -i "$mp4" \
    -map a:0 -vn \
    -c:a libmp3lame -b:a 192k -ar 44100 -ac 2 \
    -f segment -segment_time 1 -segment_start_number 1 \
    "${audiochunks_dir}/%06d.mp3" \
    && echo "  音频切片完成 -> ${audiochunks_dir}/%06d.mp3" \
    || echo "  音频切片失败（可能该视频无音轨，继续处理下一个）"

  # 其余目录保持为空，不做任何写入
done

echo "全部处理完成。"
