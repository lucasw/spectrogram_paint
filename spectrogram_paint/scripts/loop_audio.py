#!/usr/bin/env python
#coding=utf-8

import numpy as np
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

while True:
    rate, audio = wavfile.read("test.wav")
    print rate, audio.shape, np.min(audio), np.max(audio)
    scikits.audiolab.play(audio, fs=rate)

stream.stop_stream()
stream.close()
p.terminate()
