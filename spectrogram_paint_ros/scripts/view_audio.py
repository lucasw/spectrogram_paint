#!/usr/bin/env python
# Lucas Walter
# TODO(lucasw) also need rviz version

import collections
import cv2
import numpy as np
import rospy

# from audio_common_msgs.msg import AudioData
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from spectrogram_paint_ros.msg import Audio


class ViewAudio():
    def __init__(self):
        self.bridge = CvBridge()
        self.fade1 = rospy.get_param("~fade1", 0.9)
        self.fade2 = rospy.get_param("~fade2", 0.99)
        # self.buffer = collections.deque(maxlen=8192)
        self.im = np.zeros((256, 1024, 3), np.uint8)
        self.pub = rospy.Publisher("waveform", Image, queue_size=1)
        self.audio = None
        self.sub = rospy.Subscriber("audio", Audio,
                                    self.audio_callback, queue_size=1)
        self.dirty = False
        self.timer = rospy.Timer(rospy.Duration(0.05), self.update)

    def audio_callback(self, msg):
        self.audio = msg
        self.dirty = True
        # for i in range(len(msg.values)):
        #     self.buffer.append(msg.values[i])

    def update(self, event):
        if not self.dirty:
            return
        self.dirty = False
        if False:
            self.im[:, :, 1:3] = (self.im[:, :, 1:3] * self.fade1).astype(np.uint8)
            self.im[:, :, 0] = (self.im[:, :, 0] * self.fade2).astype(np.uint8)
        self.im[:, :, :] = 0
        if self.audio is not None:
            width = self.im.shape[1]
            height = self.im.shape[0]
            half_ht = height * 0.5
            # loop through entire msg?
            last_y = 0
            # TODO(lucasw) skip parameter
            for i in range(0, len(self.audio.data)):
                if i >= len(self.audio.data):
                    break
                sample = self.audio.data[i] * half_ht + half_ht
                if sample >= height:
                    sample = height - 1
                if sample < 0:
                    sample = 0
                y = int(sample)  # % height
                y0 = min(last_y, y)
                y1 = max(last_y, y)
                im_ind = i % width
                self.im[y0:y1 + 1, im_ind, 0] = 255
                self.im[y0:y1 + 1, im_ind, 1] = 10 * i  / width
                self.im[y0:y1 + 1, im_ind, 2] = 2 * i  / width
                last_y = y
        self.pub.publish(self.bridge.cv2_to_imgmsg(self.im, "bgr8"))

if __name__ == '__main__':
    rospy.init_node('view_audio')
    view = ViewAudio()
    rospy.spin()
