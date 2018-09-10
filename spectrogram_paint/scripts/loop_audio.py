#!/usr/bin/env python
#coding=utf-8

import numpy as np
import os
import scikits.audiolab
import sys
import time

from scipy.io import wavfile


chunk = 1024

if False:
    pya = pyaudio.PyAudio()
    stream = pya.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=44100,
                    output=True)

wav_name = "test.wav"
stamp = None
while True:
    new_stamp = os.stat(wav_name).st_mtime
    if new_stamp != stamp:
        stamp = new_stamp
        rate, audio = wavfile.read(wav_name)
        print wav_name, rate, audio.shape, audio.shape[0] / rate, np.min(audio), np.max(audio), stamp
    scikits.audiolab.play(audio, fs=rate)

stream.stop_stream()
stream.close()
p.terminate()
