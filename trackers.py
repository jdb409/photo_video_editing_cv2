import cv2
import rects
import utils


class Face():
    """
    Data on facial features: face, eyes, nose, mouth
    """

    def __init__(self):
        """
        Each rect has format (x,y,w,h)
        opencv does not always send in this format, so much
        check what representation is
        """

        self.face_rect = None
        self.left_eye_rect = None
        self.right_eye_rect = None
        self.nose_rect = None
        self.mouth_rect = None


class FaceTracker():
    """
    A tracker for facial features: face, eyes, nose, and mouth
    """

    def __init__(self, scale_factor=1.2, min_neighbors=2, flags=cv2.CASCADE_SCALE_IMAGE):

        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.flags = flags

        self._faces = []
        self._face_classifier = cv2.CascadeClassifier(
            'cascades/haarcascade_frontalface_alt.xml')
        self._eye_classifier = cv2.CascadeClassifier(
            'cascades/haarcascade_eye.xml')
        self._nose_classifier = cv2.CascadeClassifier(
            'cascades/haarcascade_mcs_nose.xml')
        self._mouth_classifier = cv2.CascadeClassifier(
            'cascades/haarcascade_mcs_mouth.xml')

    @property
    def faces(self):
        """Tracked faces"""
        return self._faces

    def update(self, image):
        """updates tracked face features"""

        self._faces = []

        if (utils.is_gray(image)):
            image = cv2.equalizeHist(image)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.equalizeHist(image)

        min_size = utils.widthHeightDividedBy(image, 8)

        face_rects = self._face_classifier.detectMultiScale(
            image, self.scale_factor, self.min_neighbors, self.flags, min_size)

        if (face_rects is not None):
            for face_rect in face_rects:
                face = Face()
                face.face_rect = face_rect
                x, y, w, h = face_rect

                # look for an eye in the upper-left part of the face
                search_rect = (x+w//7, y, w*2//7, h//2)
                face.left_eye_rect = self._detect_one_object(
                    self._eye_classifier, image, search_rect, 64
                )

                # look for an eye in the upper-right part of the face
                search_rect = (x + (w*4)//7, y, w*2//7, h//2)
                face.right_eye_rect = self._detect_one_object(
                    self._eye_classifier, image, search_rect, 64
                )

                # look for an node in the middle part of the face
                search_rect = (x+w//4, y+h//4, w*2, h//2)
                face.nose_rect = self._detect_one_object(
                    self._nose_classifier, image, search_rect, 32
                )

                # look for an mouth in the lower-middle part of the face
                search_rect = (x+w//6, y + (h*2)//3, w*2//3, h//3)
                face.mouth_rect = self._detect_one_object(
                    self._mouth_classifier, image, search_rect, 16
                )

                self._faces.append(face)

    def _detect_one_object(self, classifier, image, search_rect, image_size_to_min_size_ratio):
        """
        detects object in region sepcified by search_rect
        will only detect image > image_size_to_min_size_ratio
        classifier = haarscascade
        search_rect = [x,y,w,h]

        """

        x, y, w, h = search_rect

        min_size = utils.widthHeightDividedBy(
            image, image_size_to_min_size_ratio)

        sub_image = image[y:y+h, x:x+w]
        sub_rects = []
        try:
            sub_rects = classifier.detectMultiScale(
                sub_image, self.scale_factor, self.min_neighbors, self.flags, min_size)
        except Exception as e:
            print("classifer", classifier)
            print(e)

        if (len(sub_rects) == 0):
            return None

        sub_x, sub_y, sub_w, sub_h = sub_rects[0]

        return (x+sub_x, y+sub_y, sub_w, sub_h)

    def draw_debug_rects(self, image):
        """Draw rectangles around the tracked facial features."""

        if (utils.is_gray(image)):
            faceColor = 255
            leftEyeColor = 255
            rightEyeColor = 255
            noseColor = 255
            mouthColor = 255
        else:
            faceColor = (255, 255, 255)  # white
            leftEyeColor = (0, 0, 255)  # red
            rightEyeColor = (0, 255, 255)  # yellow
            noseColor = (0, 255, 0)  # green
            mouthColor = (255, 0, 0)  # blue

        for face in self.faces:
            # print(face)
            rects.outline(image, face.face_rect, faceColor)
            rects.outline(image, face.left_eye_rect, leftEyeColor)
            rects.outline(image, face.right_eye_rect,
                          rightEyeColor)
            rects.outline(image, face.nose_rect, noseColor)
            rects.outline(image, face.mouth_rect, mouthColor)
