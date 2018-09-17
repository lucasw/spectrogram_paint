#!/usr/bin/env python

import cv2
import numpy as np
import os
import pyaudio
import rospy
import time
import sys

from cv_bridge import CvBridge, CvBridgeError
from scipy import signal
from scipy.io import wavfile
from skimage import io
from sensor_msgs.msg import Image
from spectrogram_paint_ros.msg import Audio


class SpectrogramImageToAudio:
    def __init__(self):
        # TODO(lucasw) dynamic reconfigure
        self.fs = rospy.get_param("~sample_rate", 44100)
        self.onesided = rospy.get_param("~onesided", True)
        self.lowcut = rospy.get_param("~lowcut", 20)
        self.highcut = rospy.get_param("~highcut", 18000)
        self.bandpass_order = rospy.get_param("~bandpass_order", 3)
        self.do_bandpass = rospy.get_param("~do_bandpass", True)
        # TODO(lucasw) notch out 2.0-2.5 KHz?

        self.is_dirty = True
        self.bridge = CvBridge()

        self.second_pass = False
        self.mag_msg = None
        self.phase_msg = None

        self.mag_sub = rospy.Subscriber("magnitude", Image, self.mag_callback, queue_size=5)
        self.phase_sub = rospy.Subscriber("phase", Image, self.phase_callback, queue_size=5)

        self.pub = rospy.Publisher("audio", Audio, queue_size=10)

        self.mag = None

        while not rospy.is_shutdown():
            self.update()
            rospy.sleep(0.05)

    def mag_callback(self, msg):
        self.mag_msg = msg
        self.is_dirty = True

    def phase_callback(self, phase):
        self.phase_msg = msg
        self.is_dirty = True

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = signal.lfilter(b, a, data)
        return y

    def update(self, event=None):
        if self.mag_msg is None:
            return

        if self.mag is not None:
            cv2.imshow("mag", self.mag)
            cv2.waitKey(5)

        if not self.is_dirty:
            return
        self.is_dirty = False

        mag = self.bridge.imgmsg_to_cv2(self.mag_msg)

        # log scaling of image in y
        # TODO(lucasw) also rescale also?  Maybe the incoming image can be lower
        # resolution, but the remapped one can be 2048 high or some other parameter defined height?
        xr = np.arange(0.0, mag.shape[1], dtype=np.float32).reshape(1, -1)
        map_x = np.repeat(xr, mag.shape[0], axis=0)
        yr = np.arange(0.0, mag.shape[0], dtype=np.float32).reshape(-1, 1)
        # yr = 10 ** yr
        yr = yr / np.max(yr) * mag.shape[1] - 1
        yr = (np.log10(yr + 1) * np.max(yr)) / np.log10(np.max(yr) + 1)
        print yr
        map_y = np.repeat(yr, mag.shape[1], axis=1)

        self.mag = cv2.remap(mag, map_x, map_y, cv2.INTER_LINEAR)

        # want to have lower frequencies on the bottom of the image,
        # but the istft expects the opposite.
        mag = np.flipud(self.mag)

        phase = None
        if self.phase_msg is not None:
            phase = np.flipud(self.bridge.imgmsg_to_cv2(self.phase_msg))
            if phase.shape != mag.shape:
                rospy.logwarn(str(phase.shape) + '!=' + str(mag.shape))
                phase = None

        if phase is None:
            phase = mag * 0.0

        # TODO(lucasw) where did the 4 come from?
        mag = np.exp(mag * 4) - 1.0
        zxx = mag * np.exp(1j * phase)

        to, x_unfiltered = signal.istft(zxx, fs=self.fs, input_onesided=self.onesided)

        # filter out the DC and the higher frequencies
        # could lean on the producer of the image to do that though
        if self.do_bandpass:
            xo = self.butter_bandpass_filter(x_unfiltered,
                                             lowcut=self.lowcut, highcut=self.highcut,
                                             fs=self.fs, order=self.bandpass_order)
        else:
            xo = x_unfiltered

        # TODO(lucasw) notch out 2.0-2.5 KHz

        # TODO(lucasw) does this produce smoother audio?
        if self.second_pass:
            nperseg = zxx.shape[0] * 2 - 1
            print 'nperseg', nperseg
            f2, t2, zxx2 = signal.stft(xo, fs=fs, nperseg=nperseg, return_onesided=self.onesided)
            print 'max frequency', np.max(f2)
            print zxx2.shape

            t, x = signal.istft(zxx2, fs=fs, input_onesided=self.onesided)

            zmag = np.abs(zxx2)
            print zmag
            print 'z min max', np.min(zmag), np.max(zmag)
            logzmag = np.log10(zmag + 1e-10)
            logzmag -= np.min(logzmag)
            zangle = np.angle(zxx2)
        else:
            t = to
            x = xo

        # TODO(lucasw) move the normalization down stream
        # also if the max is < 1.0 then that is probably desired
        x_max = np.max(np.abs(x))
        if x_max != 0.0:
            x = x / x_max
        msg = Audio()
        msg.data = x.tolist()
        msg.sample_rate = self.fs
        self.pub.publish(msg)

if __name__ == '__main__':
    rospy.init_node('spectrogram_image_to_audio')
    spectrogram_image_to_audio = SpectrogramImageToAudio()
    rospy.spin()
