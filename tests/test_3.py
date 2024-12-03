from utils import display_video, jpeg_video_encode


if __name__ == "__main__":
    tests = ["sample_video.mp4"]
    videos_dir = "../assets/videos/"

    for i, t in enumerate(tests):
        output_path = f"./outputs/test3_{i}.mp4"

        jpeg_video_encode(videos_dir + t, output_video=output_path)
        display_video(output_path)
