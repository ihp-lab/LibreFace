import os
import glob

final_path = "/home/achaubey/Desktop/projects/data/DISFA/images"

videos_path = "/home/achaubey/Desktop/projects/data/DISFA/Videos_RightCamera"

all_videos = glob.glob(os.path.join(videos_path, "*.avi"))
all_videos.sort()

for vpath in all_videos:
	cur_v_name = ".".join(vpath.split("/")[-1].split(".")[:-1])
	cur_v_path = os.path.join(final_path, cur_v_name)
	os.makedirs(cur_v_path, exist_ok=True)

	ffmpeg_command = f"ffmpeg -i {vpath} '{cur_v_path}/{cur_v_name}_%04d.png'"
	os.system(ffmpeg_command)

