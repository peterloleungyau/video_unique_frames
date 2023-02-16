import numpy as np
import cv2 as cv
from skimage.metrics import structural_similarity as ssim

def is_different_frames(frame1, frame2, similarity_below_which_is_different):
    if frame1.shape != frame2.shape:
        return True
    if np.array_equal(frame1, frame2):
        return False
    if ssim(frame1, frame2, channel_axis = len(frame1.shape)-1) < similarity_below_which_is_different:
        return True
    return False

def video_to_unique_frames(input_video_file_name, output_name_pattern, similarity_below_which_is_different = 0.95, frames_to_skip = 0):
    cap = cv.VideoCapture(input_video_file_name)
    frame_idx = 0
    prev_frame = False
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Cannot receive frame. Exiting.")
            break
        frame_idx += 1
        # print("at frame {:d}".format(frame_idx))
        if frame_idx < frames_to_skip:
            continue
        if (prev_frame is False) or is_different_frames(prev_frame, frame, similarity_below_which_is_different):
            # new frame, output it
            output_name = output_name_pattern.format(frame_idx)
            print("New frame {:d} to {}".format(frame_idx, output_name))
            cv.imwrite(output_name, frame)
            prev_frame = frame.copy()
    cap.release()

## test one, has about 3 unique frames
#video_to_unique_frames("/home/peter/to_keep/projects/video_uniq_frames/qatar.mp4", "frame_{:05d}.png", similarity_below_which_is_different = 0.95, frames_to_skip = 70)

## the target one
video_to_unique_frames(input_video_file_name = "/home/peter/to_keep/learning/trading/chart_patterns/Become_a_Chart_Patterns_BEAST-aVGrr2hMY5o.mp4",
                       output_name_pattern = "test_chart_pattern_frames/frame_{:06d}.png",
                       similarity_below_which_is_different = 0.999,
                       frames_to_skip = 5205)

