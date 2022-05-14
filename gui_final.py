# -*- coding:utf-8 -*-

import os
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image
import cv2

from tracking import GazeTracking
import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


####################################### Global Variables #######################################

# fps_default = 15

# for Feedback Graph

smile_value_arr = []
eyes_value_arr = []
speak_value_arr = []

cnt_to_sec_arr = []
print_sec = []
# 첫 0sec
cnt_to_sec_arr.append(0)
print_sec.append(str(0) + "s")


cnt = 0
fps = 15

is_smile = False
is_normal = True
is_speak = False
is_speak_normal = True

gaze_is_wrong = False
face_is_wrong = False

count_normal = 0
count_abnormal = 0
count_face = 0

smile_abnormal = 0
smile_normal = 0
smile_is_wrong = False

speak_abnormal = 0
speak_normal = 0
speak_is_wrong = False


########################################################################################################################

### Function ###


def btncmd():
    messagebox.showinfo("Start", "Do you really want to start analysis?")
    print("Clicked Button 1!")


def fps_setting():
    global text, pop

    pop = Tk()
    pop.title("Fps Setting")
    pop.geometry("300x100")
    pop.resizable(False, False)

    pop1 = Label(pop, text="Fps Setting = ")
    pop1.grid(row=0, column=0)

    text = Entry(pop)
    text.grid(row=0, column=1)

    button = Button(pop, text="done", command=fps_display)
    button.grid(row=1, column=1)


def fps_display():
    global fps
    fps = round(float(text.get()), 2)
    messagebox.showinfo("Fps", str(int(fps)))
    pop.destroy()


def loadImage():
    """
    Image Load Function
    """
    global image
    window.filename = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                                 filetypes=(
                                                 ("all files", "*.*"), ("png files", "*.png"), ("mp4 files", "*.mp4")))

    # Label(window, text="이미지 경로 :" + window.filename).grid(row=1, column=0)
    Label(window, text="이미지 경로 :" + window.filename).place(x=500, y=500)
    image = ImageTk.PhotoImage(Image.open(window.filename))

    Label(image=image).grid(row=0, column=0)


def loadVideo():
    """
    Video Load Function
    """
    global webcam
    window.filename = filedialog.askopenfilename(initialdir="/", title="Select Video",
                                                 filetypes=(
                                                     ("all files", "*.*"), ("avi files", "*.avi"),
                                                     ("mp4 files", "*.mp4")))
    Label(window, text="비디오 경로 :" + window.filename).place(x=500, y=500)

    webcam = cv2.VideoCapture(window.filename)

    play_video()


def loadWebcam():
    """
    Webcam Camera Load Function
    """
    global cap
    Label(window).grid(row=0, column=0)

    cap = cv2.VideoCapture(0)

    play_webcam()


def play_video():
    """
    Show Video Function
    """
    ret, frame = webcam.read()

    # width, height
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, dsize=(int(w / 2), int(h / 2)))
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)

        label1.imgtk = imgtk
        label1.configure(image=imgtk)
        label1.after(10, play_video)

    else:
        webcam.release()
        return


def play_webcam():
    """
    Show Real-time Video Function
    """

    ret, frame = cap.read()

    # width, height
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    if ret:
        # Get the latest frame and convert into Image
        frame = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, dsize=(int(w / 2), int(h / 2)))
        img = Image.fromarray(frame)
        # Convert image to PhotoImage
        imgtk = ImageTk.PhotoImage(image=img)
        label1.imgtk = imgtk
        label1.configure(image=imgtk)
        # Repeat after an interval to capture continiously
        label1.after(20, play_webcam)

    else:
        webcam.release()
        return


########################################################################################################################

def doVideo():
    """
    Image Load Function
    """
    global writer
    global webcam
    global gaze
    global w, h, fps
    window.filename = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                                 filetypes=(
                                                 ("all files", "*.*"), ("png files", "*.png"), ("mp4 files", "*.mp4")))

    Label(window, text="분석 동영상 경로 :" + window.filename).place(x=500, y=500)

    webcam = cv2.VideoCapture(window.filename)
    gaze = GazeTracking()

    path1 = os.path.split(window.filename)[0]
    path2 = os.path.splitext("result_" + os.path.split(window.filename)[1])[0] + '.avi'
    # print(path1 + "\\" + path2)

    # width, height
    w = webcam.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = webcam.get(cv2.CAP_PROP_FRAME_HEIGHT)

    fps = round(webcam.get(cv2.CAP_PROP_FPS), 2)
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    writer = cv2.VideoWriter(path1 + "/" + path2, fourcc, fps, (int(w), int(h)))

    doAnalysis()


def doRealtime():
    # window1 = PanedWindow(window, relief="raised", bd=2)
    # window1.grid(expand=True)
    global writer
    global webcam
    global gaze
    global w, h, fps

    global count_face
    Label(window).grid(row=0, column=0)

    webcam = cv2.VideoCapture(0)
    gaze = GazeTracking()

    # width, height
    w = webcam.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = webcam.get(cv2.CAP_PROP_FRAME_HEIGHT)

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter("result_jieun_interview_2.avi",
                             fourcc, fps, (int(w), int(h)))

    doAnalysis()


def doAnalysis():
    global cnt
    global fps

    global is_smile
    global is_normal
    global gaze_is_wrong
    global face_is_wrong

    global count_normal
    global count_abnormal
    global count_face

    global smile_normal  # 중간에 잘못된 값은 무시
    global smile_abnormal  # 누적 값 계산
    global smile_is_wrong

    global speak_abnormal
    global speak_normal
    global speak_is_wrong

    global w, h

    # print(fps)

    ret, frame = webcam.read()

    # width, height
    w = webcam.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = webcam.get(cv2.CAP_PROP_FRAME_HEIGHT)

    if ret:

        gaze.refresh(frame)

        position = (2 * frame.shape[1] / 6, frame.shape[0] / 4, 4 * frame.shape[1] / 6, 3 * frame.shape[0] / 4)

        # frame = gaze.annotated_frame(position)

        cv2.rectangle(frame, (int(position[0]), int(position[1])), (int(position[2]), int(position[3])),
                      (150, 150, 150), 3)

        if gaze.is_face(position):
            cv2.putText(frame, 'Right Face Position!', (int(position[0]) + 5, int(position[1]) + 5), 2, 1, (150, 150, 150), 2)
            # cv2.putText(frame, 'Right Face Position!', (10, 450), 2, 2, (255, 0, 255), 2)

            if face_is_wrong:
                end_wrong = math.ceil(cnt / fps)

                minute = end_wrong // 60
                second = end_wrong % 60

                print("Face end = ", minute, "min ", second, "sec")
                face_is_wrong = False

            gaze_text = ""

            if gaze.is_blinking():
                gaze_text = "Blinking"
                count_normal = count_normal + 1  # 정상이 아니라면 count로 누적
            elif gaze.is_right():
                gaze_text = "Looking right"
                count_normal = count_normal + 1
            elif gaze.is_left():
                gaze_text = "Looking left"
                count_normal = count_normal + 1
            elif gaze.is_top():
                gaze_text = "Looking top"
                count_normal = count_normal + 1
            elif gaze.is_center_horizontal() and gaze.is_center_vertical():
                gaze_text = "Looking center"
                count_normal = 0  # 정상이면 초기화
                count_abnormal += 1
            else:
                gaze_text = "None"
                count_normal = count_normal + 1

            if gaze.is_smile() and not gaze.is_speaking():
                smile_text = "Smile"
                smile_normal = 0
                smile_abnormal = smile_abnormal + 1
            elif not gaze.is_smile() and not gaze.is_speaking():
                smile_text = "Not Smile"
                smile_normal = smile_normal + 1
            elif gaze.is_speaking():
                smile_text = "Speaking"
                smile_normal = 0
                smile_abnormal = smile_abnormal + 1
                speak_normal = 0
                speak_abnormal = speak_abnormal + 1

            cv2.putText(frame, gaze_text, (int(w/100), int(h/100 * 90)), cv2.FONT_HERSHEY_DUPLEX, 1, (147, 58, 31), 2)

            # Label(window, text="                             ").place(x=500, y=550)
            # Label(window, text=gaze_text).place(x=500, y=550)

            # left_pupil = gaze.pupil_left_coords()
            # right_pupil = gaze.pupil_right_coords()
            # cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.7,
            #             (147, 58, 31), 1)
            # cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.7,
            #             (147, 58, 31), 1)

            # Smile Text
            cv2.putText(frame, smile_text, (int(w/100), int(h/100 * 96)), cv2.FONT_HERSHEY_DUPLEX, 1, (147, 58, 31), 2)

            if count_normal >= fps:
                cv2.putText(frame, "Your gaze is wrong!", (int(w/100), int(h/100 * 10)), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 2)

                if not gaze_is_wrong:
                    start_wrong = math.floor(cnt / fps) - 1

                    minute = start_wrong // 60
                    second = start_wrong % 60

                    print("\nGaze start = ", minute, "min ", second, "sec")
                    gaze_is_wrong = True

                    count_abnormal = 0

            else:
                if gaze_is_wrong:
                    if count_abnormal > fps:
                        # choice
                        # end_wrong = math.ceil(cnt / fps)  # gaze is wrong + fps/2
                        end_wrong = math.ceil((cnt - count_abnormal) / fps)  # gaze is wrong 까지만

                        minute = end_wrong // 60
                        second = end_wrong % 60

                        print("Gaze end = ", minute, "min ", second, "sec")
                        gaze_is_wrong = False


            ##### Smile is wrong #####
            if smile_normal >= fps:
                speak_is_wrong = True
                cv2.putText(frame, "Not Smile!", (int(w/100), int(h/100 * 21)), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 2)

                if not smile_is_wrong:
                    smile_is_wrong = True
                    smile_abnormal = 0
            else:
                speak_is_wrong = True
                if smile_is_wrong:
                    if smile_abnormal > fps:
                        smile_is_wrong = False

            if speak_normal >= fps:
                if not speak_is_wrong:
                    speak_is_wrong = True
                    speak_abnormal = 0
            else:
                if speak_is_wrong:
                    if speak_abnormal > fps:
                        speak_is_wrong = False

            if gaze_is_wrong:
                eyes_value_arr.insert(cnt, 1)
            else:
                eyes_value_arr.insert(cnt, 0)

            if smile_is_wrong and speak_is_wrong:
                smile_value_arr.insert(cnt, 1)
            elif not smile_is_wrong and speak_is_wrong:
                smile_value_arr.insert(cnt, 0)
            elif not speak_is_wrong:
                smile_value_arr.insert(cnt, 2)


        else:
            # cv2.putText(frame, 'Wrong Face Position!', (10, 450), 2, 2, (255, 0, 255), 2)
            cv2.putText(frame, 'Wrong Face Position!', (int(position[0]) + 5, int(position[1]) + 5), 2, 1, (0, 0, 255), 2)
            count_face += 1

            if not face_is_wrong and count_face > fps / 2:
                # start_wrong = math.floor((cnt - count_face) / fps)
                # start_wrong = math.floor(cnt / fps)

                ########### 변경사항 ########################
                start_wrong = math.floor((cnt - fps/2) / fps)

                minute = start_wrong // 60
                second = start_wrong % 60

                print("\nFace start = ", minute, "min ", second, "sec")
                face_is_wrong = True

            smile_value_arr.insert(cnt, 0)
            eyes_value_arr.insert(cnt, 0)

        writer.write(frame)

        """
        현재 아이디어
        for문으로 cnt 일정 간격마다 (sec)로 변환해서 리스트에 넣기
        그럼 그 리스트를 바탕으로 label에 넣기 (xticks, yticks)
        """

        sec = int(cnt / fps)
        cnt = cnt + 1

        try:
            sec3 = cnt / fps
        except ZeroDivisionError:
            sec3 = 0

        # 1초마다 label 출력되도록
        if cnt % fps < 1 and (sec3 % 3) == 0:
            cnt_to_sec_arr.append(cnt)
            print_sec.append(str(sec + 1) + "s")     # 1초마다 삽입


        if cv2.waitKey(1) == 27:
            pop_graph = Tk()  # window name

            pop_graph.title("FeedBack Graph")
            pop_graph.geometry("500x500+100+100")  # fixed

            x = np.arange(0, cnt)
            y = eyes_value_arr
            y2 = smile_value_arr

            fig = Figure(figsize=(7, 10), dpi=100)

            ax1 = fig.add_subplot(2, 1, 1)
            ax2 = fig.add_subplot(2, 1, 2)
            fig.subplots_adjust(left=0.3, bottom=0.1, right=0.9, top=0.9, hspace=1)

            ax1.plot(x, y, 'r')
            ax2.plot(x, y2, 'b')

            ax1.set_title('Gaze Graph')
            ax1.set_xlabel('frame number')
            ax1.set_ylabel('status')
            ax1.set_xticks(cnt_to_sec_arr)
            ax1.set_xticklabels(print_sec)
            ax1.set_yticks([0, 1])
            ax1.set_yticklabels(["normal", "abnormal"])

            ax2.set_title('Smile Graph')
            ax2.set_xlabel('frame number')
            ax2.set_ylabel('status')
            ax2.set_xticks(cnt_to_sec_arr)
            ax2.set_xticklabels(print_sec)
            ax2.set_yticks([0, 1, 2])
            ax2.set_yticklabels(["smile", "nonsmile", "speaking"])

            canvas = FigureCanvasTkAgg(fig, master=pop_graph)
            canvas.get_tk_widget().grid(row=0, column=1)

            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, dsize=(int(w / 2), int(h / 2)))

        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)

        label1.imgtk = imgtk
        label1.configure(image=imgtk)
        label1.after(10, doAnalysis)

    else:
        if gaze_is_wrong:
            end_wrong = math.floor((cnt - count_abnormal) / fps)

            minute = end_wrong // 60
            second = end_wrong % 60

            print("Gaze end = ", minute, "min ", second, "sec")

        if face_is_wrong:
            end_wrong = math.floor(cnt / fps)

            minute = end_wrong // 60
            second = end_wrong % 60

            print("Face end = ", minute, "min ", second, "sec")

        print("!!!!!!!!!!!!!!!!!! Analysis Finished!!!!!!!!!!!!!!!!!!!!!!!!!!!")


        #### Window for Feedback Graph ####

        pop_graph = Tk()  # window name

        pop_graph.title("FeedBack Graph")
        pop_graph.geometry("500x500")  # fixed
        # pop_graph.geometry("500x500+100+100")  # fixed

        x = np.arange(0, cnt)
        y = eyes_value_arr
        y2 = smile_value_arr

        fig = Figure(figsize=(5, 5), dpi=100)

        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        fig.subplots_adjust(left=0.3, bottom=0.1, right=0.9, top=0.9, hspace=1)

        ax1.plot(x, y, 'r')
        ax2.plot(x, y2, 'b')

        ax1.set_title('Gaze Graph')
        ax1.set_xlabel('frame number')
        ax1.set_ylabel('status')
        ax1.set_xticks(cnt_to_sec_arr)
        ax1.set_xticklabels(print_sec)
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(["normal", "abnormal"])

        ax2.set_title('Smile Graph')
        ax2.set_xlabel('frame number')
        ax2.set_ylabel('status')
        ax2.set_xticks(cnt_to_sec_arr)
        ax2.set_xticklabels(print_sec)
        ax2.set_yticks([0, 1, 2])
        ax2.set_yticklabels(["smile", "nonsmile", "speaking"])

        canvas = FigureCanvasTkAgg(fig, master=pop_graph)
        canvas.get_tk_widget().grid(row=0, column=1)

        # extent = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        # fig.savefig('ax1_figure.png', bbox_inches=extent)

        # extent = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        # fig.savefig('ax2_figure.png', bbox_inches=extent)

        return


########################################################################################################################

# def do(*args):

#     if variable.get() == "Video Analysis":
#         doVideo()
#     elif variable.get() == "Fps Setting":
#         fps_setting()
#     elif variable.get() == "Exit":
#         window.quit()

########################################################################################################################


if __name__ == "__main__":
    ### Window ###
    window = Tk()  # window name

    window.title("Online Interview Test Program")
    window.geometry("960x540")  # fixed
    window.resizable(False, False)

    ### Label ###
    label1 = Label(window, text="Welcome to our graduation project")
    label1.grid(row=0, column=0)

    ### Menu ###
    menubar = Menu(window)

    ### File menu
    menu1 = Menu(menubar, tearoff=0)
    menubar.add_cascade(label="File", menu=menu1)
    menu1.add_command(label="Load Image", command=loadImage)
    menu1.add_command(label="Load Video", command=loadVideo)
    menu1.add_separator()
    menu1.add_command(label="Real Time", command=loadWebcam)

    menu1 = Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Analysis", menu=menu1)
    menu1.add_command(label="Video analysis", command=doVideo)
    menu1.add_command(label="Realtime analysis", command=doRealtime)
    
    menu2 = Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Option", menu=menu2)
    menu2.add_command(label="Fps setting", command=fps_setting)
    menu2.add_command(label="Exit", command=window.quit)
    
    window.config(menu=menubar)

    window.mainloop()