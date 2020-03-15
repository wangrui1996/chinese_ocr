import os
video_path = "/Users/wangrui/Downloads/111"

video_list = os.listdir(video_path)



for video in video_list:
    if video.split(".")[1] not in ['mp4']:
        print(video)
        continue
    save_video = os.path.join(video_path, "{}a.mp4".format(video.split(".")[0]))

    cmd = "ffmpeg -i {} -c:v libx264  {}".format(os.path.join(video_path, video), save_video)
    print(cmd)
    os.system(cmd)
