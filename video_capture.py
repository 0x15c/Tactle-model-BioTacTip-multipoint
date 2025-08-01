import numpy as np
import cv2 as cv
import time



output_video_file_path = 'output_from_online_cap.mp4'
cap = cv.VideoCapture(0)
fps = cap.get(cv.CAP_PROP_FPS)
video_w, video_h = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fourcc = cv.VideoWriter_fourcc(*'XVID')  # Codec for MP4

# crop and preprocessing
radius = 160
center = (175, 175)
cnt = [int(video_w/2),int(video_h/2)]
crop_px = 175
crop_py = 175
crop_offset_x = 0
crop_offset_y = -8
cropped_limits = [[cnt[0]-crop_px+crop_offset_x,cnt[1]-crop_py+crop_offset_y],[cnt[0]+crop_px+crop_offset_x,cnt[1]+crop_py+crop_offset_y]]
cropped_size = [2*crop_px, 2*crop_py]

# camera settings
cap.set(cv.CAP_PROP_FRAME_WIDTH, video_w)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, video_h)
cap.set(cv.CAP_PROP_EXPOSURE, -7.5)

# output video file settings
out_original = cv.VideoWriter('output_from_online_cap.mp4', fourcc, fps, cropped_size, True)

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video stream or error reading frame.")
        break
    cropped_frame = frame[cropped_limits[0][1]:cropped_limits[1][1],cropped_limits[0][0]:cropped_limits[1][0]]
    grey_frame = cv.cvtColor(cropped_frame,cv.COLOR_BGR2GRAY)
    cv.imshow('foo',grey_frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    # write capture
    bgr_image = cv.cvtColor(grey_frame, cv.COLOR_GRAY2BGR)
    out_original.write(bgr_image)
cap.release()
out_original.release()
cv.destroyAllWindows()