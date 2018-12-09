import cv2
from managers import WindowManager, CaptureManager
import filters
import rects
from trackers import FaceTracker


class Cameo():

    def __init__(self):
        self._windowManager = WindowManager('Cameo', self.onKeypress)

        self._captureManager = CaptureManager(
            cv2.VideoCapture(0), self._windowManager, True)

        self._curveFilter = filters.BGRPortraCurveFilter()
        self._conv_filter = filters.SharpenFilter()
        self._face_tracker = FaceTracker()
        self._should_draw_debug_rects = False

    def run(self):
        """
        run main loop
        """
        self._windowManager.createWindow()
        while (self._windowManager.isWindowCreated):
            self._captureManager.enterFrame()
            frame = self._captureManager.frame

            # filters.strokeEdges(frame, frame)
            # self._conv_filter.apply(frame, frame)

            self._face_tracker.update(frame)
            faces = self._face_tracker.faces
            rects.swap_rects(frame, frame, [face.face_rect for face in faces])

            if self._should_draw_debug_rects:
                self._face_tracker.draw_debug_rects(frame)

            self._captureManager.exitFrame()
            self._windowManager.processEvents()

    def onKeypress(self, keycode):
        """Handle a keypress.

        space -> Take a screenshot.
        tab -> Start/stop recording a screencast.
        escape -> Quit.
        x ->start/stop drawing debug rects

        """
        if (keycode == 32):  # space
            self._captureManager.writeImage('output/screenshot.png')
        elif (keycode == 9):  # tab
            if (not self._captureManager.isWritingVideo):
                self._captureManager.startWritingVideo('output/screencast.avi')
            else:
                self._captureManager.stopWritingVideo()
        elif keycode == 120:  # x
            self._should_draw_debug_rects = not self._should_draw_debug_rects
        elif (keycode == 27):  # escape
            self._windowManager.destroyWindow()


if __name__ == "__main__":
    Cameo().run()
