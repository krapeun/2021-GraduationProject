#-*- coding:utf-8 -*-

import cv2
from tracking import GazeTracking
from matplotlib import pyplot as plt
import numpy as np


video = "jieun_lips.mp4"
video_name = video.split(".")[0]

webcam = cv2.VideoCapture("video/smile/{}".format(video))
print(webcam.get(cv2.CAP_PROP_POS_MSEC))

frame_width = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = round(webcam.get(cv2.CAP_PROP_FPS), 2)

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
# writer = cv2.VideoWriter("video/output/result_{}.cv2.CAP_PROP_FPSavi".format(video_name), fourcc, 24, (frame_width, frame_height))
writer = cv2.VideoWriter("video/smile/output/result_{}.avi".format(video_name), fourcc, fps, (frame_width, frame_height))
print(fps)

gaze = GazeTracking()

# frame number
cnt = 0

# judge class
is_smile = False
is_normal = True
count_normal = 0    # normal 누적이 25frame 이상이라면 is_normal = False

smile_value_arr = []
eyes_value_arr = []

while webcam.isOpened():

    ret, frame = webcam.read()

    if ret:

        gaze.refresh(frame)

        position = (2 * frame.shape[1] / 6, frame.shape[0] / 4, 4 * frame.shape[1] / 6, 3 * frame.shape[0] / 4)

        frame = gaze.annotated_frame(position)

        cv2.rectangle(frame, (int(position[0]), int(position[1])), (int(position[2]), int(position[3])), (102, 255, 255), 3)

        if gaze.is_face(position):
            cv2.putText(frame, 'Right Face Position!', (10, 450), 2, 2, (255, 0, 255), 2)

            text1 = ""
            text2 = ""

            if gaze.is_blinking():
                text = "Blinking"
                count_normal = count_normal + 1     # 정상이 아니라면 count로 누적
                cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

            else:
                if gaze.is_right():
                    text1 = "Looking right"
                    count_normal = count_normal + 1
                elif gaze.is_left():
                    text1 = "Looking left"
                    count_normal = count_normal + 1
                elif gaze.is_center_horizontal():
                    text1 = "Looking center"
                    count_normal = 0    # 정상이면 초기화

                if gaze.is_top():
                    text2 = ", top"
                    count_normal = count_normal + 1
                elif gaze.is_bottom():
                    text2 = ", bottom"
                    count_normal = count_normal + 1
                elif gaze.is_center_vertical():
                    text2 = ", center"
                else:
                    text2 = "None"
                    count_normal = count_normal + 1


            ### Facial Expression ###

            if gaze.is_smile() and not gaze.is_speaking():
                smile_text = "Smile"
                smile_value = 1
            elif not gaze.is_smile() and not gaze.is_speaking():
                smile_text = "Not Smile"
                smile_value = 0
            elif gaze.is_speaking():
                smile_text = "Speaking"

            smile_value_arr.insert(cnt, smile_value)
            eyes_value_arr.insert(cnt, count_normal)

            cv2.putText(frame, text1 + text2, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

            left_pupil = gaze.pupil_left_coords()
            right_pupil = gaze.pupil_right_coords()
            cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.7, (147, 58, 31), 1)
            cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.7, (147, 58, 31), 1)

            # Smile Text
            cv2.putText(frame, smile_text, (90, 200), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)

            if count_normal >= fps:
                cv2.putText(frame, "Your gaze is wrong!", (90, 250), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 1)

        else:
            cv2.putText(frame, 'Wrong Face Position!', (10, 450), 2, 2, (255, 0, 255), 2)

        writer.write(frame)

        cv2.imshow('demo', frame)

        sec = int(cnt / fps)
        cv2.imwrite("labeling/frame/jieun/frame{}_{}.jpg".format(sec, cnt), frame)
        cnt = cnt + 1

        if cv2.waitKey(1) == 27:
            break

    else:
        break

webcam.release()
writer.release()
cv2.destroyAllWindows


### Smile Graph ####
# x = np.arange(0, cnt)
# y = smile_value_arr
# plt.ylim(-1, 2)
# plt.plot(x, y)
# plt.xlabel('frame number')
# plt.ylabel('smile')
# plt.title('smile graph')
# plt.show()


# ### Eyes Graph ####
# x = np.arange(0, cnt)
# y = eyes_value_arr
# plt.plot(x, y)
# plt.xlabel('frame number')
# plt.ylabel('normal')
# plt.title('eyes graph')
# plt.show()

### Smile Graph ####
x = np.arange(0, cnt)
y = smile_value_arr
# plt.ylim(-1, 2)
# plt.plot(x, y)
# plt.xlabel('frame number')
# plt.ylabel('smile')
# plt.title('smile graph')
# plt.show()

# plt.plot(x,y,'b',label='first')
# plt.plot(x,y2,'r',label='second')
# plt.legend(loc='upper right')


### Eyes Graph ####
x = np.arange(0, cnt)
y2 = eyes_value_arr
plt.plot(x, y)
plt.plot(x, y2)

plt.xlabel('frame number')
plt.ylabel('normal')
plt.title('eyes graph')
plt.show()
