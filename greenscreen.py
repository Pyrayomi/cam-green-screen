import cv2
import numpy as np
import pyvirtualcam

class GreenScreen:
    def __init__(self, webcam=0, color=(0, 255, 0), kernel_type=0, kernel_size=(30, 30), noisey_kernel=(3, 3), thresh=30):
        self.webcam = webcam
        self.color = color
        self.kernel_type = kernel_type
        self.kernel_size = kernel_size
        self.noisey_kernel = noisey_kernel
        self.thresh = thresh
        self.debug = False
        self.show_webcam = False
        self.exit_key = 27  # ESC
        self.reset_key = 13  # Enter

    def find_dif(self, orig, img):
        diff = cv2.absdiff(orig, img)  # Find the difference
        diff[diff < self.thresh] = 0  # Remove the difference if it meets the threshold
        mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)  # Make it black and white

        # Noise reduction, circle seems best for this, no matter the circumstances
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.noisey_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # "Blob" reduction (the false positives inside the changed images)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.kernel_size)
        if self.kernel_type == 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, self.kernel_size)
        elif self.kernel_type == 2:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.kernel_size)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        fgmask = mask.astype(np.uint8)
        fgmask[fgmask > 0] = 255  # Convert to only white and black
        return fgmask

    def run_virtual_camera(self):
        cap = cv2.VideoCapture(self.webcam)
        ret, orig = cap.read()

        grn_screen = np.zeros(orig.shape, np.uint8)  # An array of bytes the same size as our image
        grn_screen[:] = self.color  # Make all those bytes a color (green! blue! purple! who cares!)

        with pyvirtualcam.Camera(width=orig.shape[1], height=orig.shape[0], fps=20, fmt=pyvirtualcam.PixelFormat.BGR) as cam:
            while True:
                ret, frame = cap.read()
                fgmask = self.find_dif(orig, frame)

                # Greenscreen creation
                bgmask = cv2.bitwise_not(fgmask)  # Invert foreground
                fgimg = cv2.bitwise_and(frame, frame, mask=fgmask)  # Cut out the foreground
                bgimg = cv2.bitwise_and(grn_screen, grn_screen, mask=bgmask)  # Cut out the background
                wgs = cv2.add(fgimg, bgimg)  # Combine each cut

                if self.show_webcam:
                    wgs = frame

                cam.send(wgs)
                cam.sleep_until_next_frame()

                if self.debug:
                    cv2.imshow('orig', frame)
                    cv2.imshow('mask', fgmask)
                cv2.imshow('result', wgs)

                k = cv2.waitKey(25) & 0xff
                if k == self.exit_key:
                    break
                if k == self.reset_key:
                    ret, orig = cap.read()

            cap.release()
            cv2.destroyAllWindows()

def main():
    greenscreen = GreenScreen()
    greenscreen.run_virtual_camera()

if __name__ == "__main__":
    main()
