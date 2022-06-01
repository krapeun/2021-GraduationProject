from __future__ import division
import cv2
import dlib
import math

from tracking.facial_landmark import FacialLandmark
from .eye import Eye
from .calibration import Calibration


class GazeTracking(object):

    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = Calibration()

        #### 추가 사항 ####
        self.facial_landmark = None
        self.face = None

        self._face_detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor("./tracking/trained_models/shape_predictor_68_face_landmarks.dat")

    @property
    def pupils_located(self):
        """Check that the pupils have been located"""
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False

    def refresh(self, frame):
        """Refreshes the frame and analyzes it"""
        self.frame = frame
        self._analyze()

    def _analyze(self):
        """ Detects the face and initialize Eye objects"""
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector(frame)

        if faces:
            self.face = faces[0]

        try:
            landmarks = self._predictor(frame, faces[0])

            self.eye_left = Eye(frame, landmarks, 0, self.calibration)
            self.eye_right = Eye(frame, landmarks, 1, self.calibration)

            self.facial_landmark = FacialLandmark(frame, landmarks)

        except IndexError:
            self.eye_left = None
            self.eye_right = None

    def pupil_left_coords(self):
        """Returns the coordinates of the left pupil"""
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return (x, y)

    def pupil_right_coords(self):
        """Returns the coordinates of the right pupil"""
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return (x, y)

    def horizontal_ratio(self):
        if self.pupils_located and self.eye_left.blinking_coords and self.eye_right.blinking_coords:
            ### left eye ###
            left_x = self.eye_left.origin[0] + self.eye_left.pupil.x
            left_y = self.eye_left.origin[1] + self.eye_left.pupil.y

            left_left = math.hypot((left_x - self.eye_left.blinking_coords[0][0]),
                                   (left_y - self.eye_left.blinking_coords[0][1]))
            left_right = math.hypot((self.eye_left.blinking_coords[1][0] - left_x),
                                    (self.eye_left.blinking_coords[1][1] - left_y))

            try:
                pupil_left = left_left / left_right
            except ZeroDivisionError:
                return 0

            ### right eye ###
            right_x = self.eye_right.origin[0] + self.eye_right.pupil.x
            right_y = self.eye_right.origin[1] + self.eye_right.pupil.y

            right_left = math.hypot((right_x - self.eye_right.blinking_coords[0][0]),
                                    (right_y - self.eye_right.blinking_coords[0][1]))
            right_right = math.hypot((self.eye_right.blinking_coords[1][0] - right_x),
                                     (self.eye_right.blinking_coords[1][1] - right_y))

            try:
                pupil_right = right_left / right_right
            except ZeroDivisionError:
                return 0

            return (pupil_left + pupil_right) / 2

    def vertical_ratio(self):
        if self.pupils_located and self.eye_left.blinking_coords and self.eye_right.blinking_coords:
            ### left eye ###
            left_x = self.eye_left.origin[0] + self.eye_left.pupil.x
            left_y = self.eye_left.origin[1] + self.eye_left.pupil.y

            left_top = math.hypot((left_x - self.eye_left.blinking_coords[2][0]),
                                  (left_y - self.eye_left.blinking_coords[2][1]))
            left_bottom = math.hypot((self.eye_left.blinking_coords[3][0] - left_x),
                                     (self.eye_left.blinking_coords[3][1] - left_y))

            try:
                pupil_left = left_top / left_bottom
            except ZeroDivisionError:
                return 0

            ### right eye ###
            right_x = self.eye_right.origin[0] + self.eye_right.pupil.x
            right_y = self.eye_right.origin[1] + self.eye_right.pupil.y

            right_top = math.hypot((right_x - self.eye_right.blinking_coords[2][0]),
                                   (right_y - self.eye_right.blinking_coords[2][1]))
            right_bottom = math.hypot((self.eye_right.blinking_coords[3][0] - right_x),
                                      (self.eye_right.blinking_coords[3][1] - right_y))

            try:
                pupil_right = right_top / right_bottom
            except ZeroDivisionError:
                return 0

            return (pupil_left + pupil_right) / 2

    def IoU(self, box1, box2):
        # box = (x1, y1, x2, y2)
        # obtain x1, y1, x2, y2 of the intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # compute the width and height of the intersection
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)

        inter = w * h
        iou = inter

        return iou

    def is_face(self, box):
        if self.face:
            face = (self.face.left(), self.face.top(), self.face.right(), self.face.bottom())
            size = (self.face.right() - self.face.left()) * (self.face.bottom() - self.face.top())

            percent = round((self.IoU(box, face) / size) * 100, 2)

            # return percent
            return percent >= 80

    def is_right(self):
        """Returns true if the user is looking to the right"""
        if self.pupils_located:
            return self.horizontal_ratio() <= 0.6

    def is_left(self):
        """Returns true if the user is looking to the left"""
        if self.pupils_located:
            return self.horizontal_ratio() >= 1.4

    def is_center_horizontal(self):
        """Returns true if the user is looking to the center"""
        if self.pupils_located:
            return self.is_right() is not True and self.is_left() is not True

    def is_top(self):
        """Returns true if the user is looking to the top"""
        if self.pupils_located:
            return self.vertical_ratio() <= 0.4

    def is_bottom(self):
        """Returns true if the user is looking to the bottom"""
        if self.pupils_located:
            return self.vertical_ratio() >= 1.6

    def is_center_vertical(self):
        """Returns true if the user is looking to the center"""
        if self.pupils_located:
            return self.is_top() is not True and self.is_bottom() is not True

    def is_blinking(self):
        """ Returns true if the user closes his eyes"""
        if self.eye_left is not None and self.eye_right is not None:
            blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2
        else:
            return False

        return blinking_ratio < 0.23

    def annotated_frame(self, box):
        """Returns the main frame with pupils highlighted"""
        frame = self.frame.copy()

        if self.face:
            cv2.rectangle(frame, (self.face.left(), self.face.top()), (self.face.right(), self.face.bottom()), (0, 0, 255), 3)

        if self.is_face(box):
            if self.pupils_located:
                color = (0, 255, 0)
                x_left, y_left = self.pupil_left_coords()
                x_right, y_right = self.pupil_right_coords()
                cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
                cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
                cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
                cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)

        return frame

    ########### Facial Landmark ###########

    def is_speaking(self):
        """ Return true if the user speaks """
        if self.pupils_located:
            speak_ratio = self.facial_landmark.speak
            return speak_ratio > 0.1

    def is_smile(self):
        """ Return true if the user smiles """
        if self.pupils_located:
            smile_ratio = self.facial_landmark.smile
            return smile_ratio > 1.0

