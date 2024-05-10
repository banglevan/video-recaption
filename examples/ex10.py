# import av
import cv2
import json
import os
# import shutil
import sys
# import subprocess
import time
# from decimal import Decimal
# from decord import VideoReader
# from ffms2 import VideoSource
# from moviepy.editor import VideoFileClip
from typing import List


def with_movie_py(video: str) -> List[int]:
    """
    Link: https://pypi.org/project/moviepy/
    My comments:
        The timestamps I get are not good compared to gMKVExtractGUI or ffms2.
        (I only tried with VFR video)

    Parameters:
        video (str): Video path
    Returns:
        List of timestamps in ms
    """
    vid = VideoFileClip(video)

    timestamps = [
        round(tstamp * 1000) for tstamp, frame in vid.iter_frames(with_times=True)
    ]

    return timestamps


def with_cv2(video: str) -> List[int]:
    """
    Link: https://pypi.org/project/opencv-python/
    My comments:
        I don't know why, but the last 4 or 5 timestamps are equal to 0 when they should not.
        Also, cv2 is slow. It took my computer 132 seconds to process the video.


    Parameters:
        video (str): Video path
    Returns:
        List of timestamps in ms
    """
    timestamps = []
    cap = cv2.VideoCapture(video)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Total frames in video: {frame_count}")
    print(f"Frames per second (FPS): {fps}")
    n = 0
    while cap.isOpened():
        frame_exists, curr_frame = cap.read()
        if frame_exists:
            n += 1
            timestamps.append(round(cap.get(cv2.CAP_PROP_POS_MSEC)))
            located = n / fps
            convert = time.strftime("%H:%M:%S", time.gmtime(located))
            h, m, s = convert.strip().split(':')
            ms = (located - (float(h) * 60 * 60 + float(m) * 60 + float(s))) * 1000
            timestamp = convert + ',' + f'{int(ms):03d}'
            print(timestamp)
        else:
            break

    cap.release()

    return timestamps


def with_pyffms2(video: str) -> List[int]:
    """
    Link: https://pypi.org/project/ffms2/
    My comments:
        Works really well, but it doesn't install ffms2 automatically,
        so you need to do it by yourself.
        The easiest way is to install Vapoursynth and use it to install ffms2.
        Also, the library doesn't seems to be really maintained.

    Parameters:
        video (str): Video path
    Returns:
        List of timestamps in ms
    """
    video_source = VideoSource(video)

    # You can also do: video_source.track.timecodes
    timestamps = [
        int(
            (frame.PTS * video_source.track.time_base.numerator)
            / video_source.track.time_base.denominator
        )
        for frame in video_source.track.frame_info_list
    ]

    return timestamps


def with_decord(video: str) -> List[int]:
    """
    Link: https://github.com/dmlc/decord
    My comments:
        Works really well, but it seems to only work with mkv and mp4 file.
        Important, Decord seems to automatically normalise the timestamps
        which can cause many issue: https://github.com/dmlc/decord/issues/238
        Mp4 file can have a +- 1 ms difference with ffms2, but it is acceptable.

    Parameters:
        video (str): Video path
    Returns:
        List of timestamps in ms
    """
    vr = VideoReader(video)

    timestamps = vr.get_frame_timestamp(range(len(vr)))
    timestamps = (timestamps[:, 0] * 1000).round().astype(int).tolist()

    return timestamps


def with_pyav(video: str, index: int = 0) -> List[int]:
    """
    Link: https://pypi.org/project/av/
    My comments:
        Works really well, but it is slower than ffprobe.
        The big advantage is that ffmpeg does not have to be installed on the computer,
        because pyav installs it automatically

    Parameters:
        video (str): Video path
        index (int): Stream index of the video.
    Returns:
        List of timestamps in ms
    """
    container = av.open(video)
    video = container.streams.get(index)[0]

    if video.type != "video":
            raise ValueError(
                f'The index {index} is not a video stream. It is an {video.type} stream.'
            )

    av_timestamps = [
        int(packet.pts * video.time_base * 1000) for packet in container.demux(video) if packet.pts is not None
    ]

    container.close()
    av_timestamps.sort()

    return av_timestamps


def with_ffprobe(video_path: str, index: int = 0) -> List[int]:
    """
    Link: https://ffmpeg.org/ffprobe.html
    My comments:
        Works really well, but the user need to have FFMpeg in his environment variables.

    Parameters:
        video (str): Video path
        index (int): Index of the stream of the video
    Returns:
        List of timestamps in ms
    """

    def get_pts(packets) -> List[int]:
        pts: List[int] = []

        for packet in packets:
            pts.append(int(Decimal(packet["pts_time"]) * 1000))

        pts.sort()
        return pts

    # Verify if ffprobe is installed
    if shutil.which("ffprobe") is None:
        raise Exception("ffprobe is not in the environment variable.")

    # Getting video absolute path and checking for its existance
    if not os.path.isabs(video_path):
        dirname = os.path.dirname(os.path.abspath(sys.argv[0]))
        video_path = os.path.join(dirname, video_path)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f'Invalid path for the video file: "{video_path}"')

    cmd = f'ffprobe -select_streams {index} -show_entries packet=pts_time:stream=codec_type "{video_path}" -print_format json'
    ffprobeOutput = subprocess.run(cmd, capture_output=True, text=True)
    ffprobeOutput = json.loads(ffprobeOutput.stdout)

    if len(ffprobeOutput) == 0:
        raise Exception(
            f"The file {video_path} is not a video file or the file does not exist."
        )

    if len(ffprobeOutput["streams"]) == 0:
        raise ValueError(f"The index {index} is not in the file {video_path}.")

    if ffprobeOutput["streams"][0]["codec_type"] != "video":
        raise ValueError(
            f'The index {index} is not a video stream. It is an {ffprobeOutput["streams"][0]["codec_type"]} stream.'
        )

    return get_pts(ffprobeOutput["packets"])


def main():
    video = r"C:\BANGLV\video_recaption\data\ex1.mp4"

    # start = time.process_time()
    # movie_py_timestamps = with_movie_py(video)
    # print(f"With Movie py {time.process_time() - start} seconds")

    start = time.process_time()
    cv2_timestamps = with_cv2(video)
    print(f"With cv2 {time.process_time() - start} seconds")

    # start = time.process_time()
    # ffms2_timestamps = with_pyffms2(video)
    # print(f"With ffms2 {time.process_time() - start} seconds")
    #
    # start = time.process_time()
    # decord_timestamps = with_decord(video)
    # print(f"With decord {time.process_time() - start} seconds")
    #
    # start = time.process_time()
    # av_timestamps = with_pyav(video)
    # print(f"With av {time.process_time() - start} seconds")
    #
    # start = time.process_time()
    # ffprobe_timestamps = with_ffprobe(video)
    # print(f"With ffprobe {time.process_time() - start} seconds")


if __name__ == "__main__":
    main()