#!/usr/bin/env python
#coding=utf-8

import numpy as np
import os
import rospy
import scikits.audiolab
import sys
import time

from scipy.io import wavfile
from spectrogram_paint_ros.msg import Audio
from threading import Lock


class LoopAudio:
    def __init__(self):
        self.mutex = Lock()
        self.audio = None
        self.audio_msg = None
        self.sub = rospy.Subscriber("audio", Audio, self.audio_callback, queue_size=4)

        while not rospy.is_shutdown():
            if self.audio is None:
                rospy.sleep(0.1)
                continue
            # print wav_name, rate, audio.shape, audio.shape[0] / rate, np.min(audio), np.max(audio), stamp
            # TODO(lucasw) mutexes, dual buffers
            # This needs to run in separate thread because it is blocking the audio callback,
            # for now sleep
            rospy.loginfo(str(self.audio.shape))
            self.mutex.acquire()
            # TODO(lucasw) deepcopy?
            audio = self.audio
            sample_rate = self.audio_msg.sample_rate
            self.mutex.release()
            scikits.audiolab.play(audio, fs=sample_rate)
            rospy.sleep(0.05)

    def audio_callback(self, msg):
        rospy.loginfo("new audio")
        # self.audio_msg = msg
        audio = np.array(msg.data)
        # this will throw if the length isn't even
        if msg.stereo:
            # TODO(lucasw) need locks to avoid problems with play() above
            audio = np.append(audio, msg.data_right)
            audio = audio.reshape((2, -1))
        self.mutex.acquire()
        self.audio = audio
        self.audio_msg = msg
        self.mutex.release()

if __name__ == '__main__':
    rospy.init_node('loop_audio')
    loop_audio = LoopAudio()
    rospy.sleep()
