#!/usr/bin/env python
import cv2
import numpy as np
import skimage.io


class Paint:
    def __init__(self):
        # Create a black image, a window and bind the function to window
        wd = 256
        ht = 800
        self.img = np.zeros((ht, wd, 1), np.float32)
        self.phase = np.zeros((ht, wd, 1), np.float32)
        skimage.io.imsave("paint_mag.tif", self.img)
        skimage.io.imsave("paint_angle.tif", self.phase)
        cv2.namedWindow('image')
        self.l_button_down = False
        self.r_button_down = False
        cv2.setMouseCallback('image', self.draw_circle)

        print self.img.shape
        while(1):
            cv2.imshow('image', self.img)
            key = cv2.waitKey(20) & 0xFF
            if key is 255:
                continue
            print key
            if key == 27 or key == ord('q'):
                print('quitting')
                break
            if key == ord('h'):
                print np.min(self.img), np.max(self.img)
            if key == ord('c'):
                print("clipping")
                self.img[self.img > 1.0] = 1.0
            if key == ord('n'):
                print("normalizing")
                im_max = np.max(self.img)
                if im_max > 0:
                    self.img /= im_max
            if key == ord('s'):
                print("updating output magnitude image")
                skimage.io.imsave("paint_mag.tif", self.img)
        cv2.destroyAllWindows()

    # mouse callback function
    def draw_circle(self, event, x, y, flags, param):
        # print event, x, y, flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            print (1.0 - float(y) / self.img.shape[0]) * 44100 / 2.0, 'Hz'
            self.l_button_down = True
        if event == cv2.EVENT_LBUTTONUP:
            self.l_button_down = False
        if event == cv2.EVENT_RBUTTONDOWN:
            self.r_button_down = True
        if event == cv2.EVENT_RBUTTONUP:
            self.r_button_down = False

        if self.l_button_down:
            for i in range(16):
                for j in range(16):
                    xi = x + i - 8
                    yi = y + j - 8
                    xi %= self.img.shape[1]
                    yi %= self.img.shape[0]
                    self.img[yi, xi] += 0.01
                    self.img[yi, xi] *= 1.05
        if self.r_button_down:
            for i in range(16):
                for j in range(16):
                    xi = x + i - 8
                    yi = y + j - 8
                    xi %= self.img.shape[1]
                    yi %= self.img.shape[0]
                    self.img[yi, xi] *= 0.97
if __name__ == '__main__':
    paint = Paint()
