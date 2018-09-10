#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import os
import pyaudio
import time
import sys

from scipy import signal
from scipy.io import wavfile
from skimage import io


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

onesided = True

# example stft output
if False:
    fs = 1024
    N = 30*fs
    nperseg = 1023
    amp = 2 * np.sqrt(2)
    noise_power = 0.001 * fs / 2
    t = np.arange(N) / float(fs)
    carrier = amp * np.sin(2*np.pi*50*t)
    noise = np.random.normal(scale=np.sqrt(noise_power),
                             size=t.shape)
    x = carrier + noise

    f, t, Zxx = signal.stft(x, fs=fs, nperseg=nperseg, return_onesided=onesided)
    zmag = np.abs(Zxx)
    zphase = np.angle(Zxx)
    print Zxx.shape, Zxx[1,1], zmag[1,1], zphase[1,1]
    print np.min(zmag), np.max(zmag)
    print np.min(zphase), np.max(zphase)

mag_file = sys.argv[1]
phase_file = sys.argv[2]
fs = 44100
loop = True
second_pass = False

do_bandpass = True
lowcut = 20
highcut = 5e3

mag_stamp = None
phase_stamp = None
mag = None
phase = None
old_mag = None
old_phase = None

while True:
    time.sleep(0.1)

    try:
        new_mag_stamp = os.stat(mag_file).st_mtime
        new_phase_stamp = os.stat(phase_file).st_mtime
    except OSError as ex:
        print(ex)
        continue
    if new_mag_stamp == mag_stamp and new_phase_stamp == phase_stamp:
        continue
    mag_stamp = new_mag_stamp
    phase_stamp = new_phase_stamp

    try:
        start = time.time()
        mag = np.flipud(io.imread(mag_file))

        if len(mag.shape) > 2:
            print 'warning mag image has multiple layers', mag.shape[2]
            # TODO(lucasw) maybe should sum the layers together?
            mag = mag[:,:,0]

        phase = np.flipud(io.imread(phase_file))
        if len(phase.shape) > 2:
            print 'warning phase image has multiple layers', phase.shape[2]
            phase = phase[:,:,0]
        end = time.time()
        print end - start
    except ValueError as ex:
        print(ex)
        continue

    if phase.shape != mag.shape:
        print phase.shape, '!=', mag.shape
        continue

    # don't need to check for differences, trust that if the modified timestamp
    # is different then do the update
    if False:
        update = old_phase is None or old_mag is None
        if not update:
            update = mag.shape != old_mag.shape
        if not update:
            update = not np.array_equal(phase, old_phase) or not np.array_equal(mag, old_mag)
        if not update:
            continue
    print 'updating', mag_stamp, phase_stamp
    print 'mag', mag_file, mag.shape, np.min(mag), np.max(mag)
    print 'phase', phase_file, phase.shape, np.min(phase), np.max(phase)
    old_mag = mag
    old_phase = phase

    mag = np.exp(mag * 4) - 1.0
    zxx = mag * np.exp(1j * phase)

    to, x_unfiltered = signal.istft(zxx, fs=fs, input_onesided=onesided)

    # filter out the DC and the higher frequencies
    if do_bandpass:
        xo = butter_bandpass_filter(x_unfiltered, lowcut=lowcut, highcut=highcut, fs=fs, order=3)
    else:
        xo = x_unfiltered

    if second_pass:
        nperseg = zxx.shape[0] * 2 - 1
        print 'nperseg', nperseg
        f2, t2, zxx2 = signal.stft(xo, fs=fs, nperseg=nperseg, return_onesided=onesided)
        print 'max frequency', np.max(f2)
        print zxx2.shape

        t, x = signal.istft(zxx2, fs=fs, input_onesided=onesided)

        zmag = np.abs(zxx2)
        print zmag
        print 'z min max', np.min(zmag), np.max(zmag)
        logzmag = np.log10(zmag + 1e-10)
        logzmag -= np.min(logzmag)
        zangle = np.angle(zxx2)
    else:
        t = to
        x = xo

    x_max = np.max(np.abs(x))
    if x_max != 0.0:
        x = x / x_max
    wavfile.write("test.wav", fs, x)

if False:
    sp = np.fft.fft(x)
    half_ind = sp.shape[-1] / 2
    sp_mag = np.abs(sp[:half_ind])
    print 'fft min max', np.min(sp_mag), np.max(sp_mag)
    freq = np.fft.fftfreq(t.shape[-1], 1.0 / fs)
    freq = freq[:half_ind]

if False:
    fig, axes = plt.subplots(3, 1, num=1)  # sharedy=True) # figure(num=0)
    # print axes.shape
    axes[0].pcolormesh(t2, f2, logzmag, vmin=0, vmax=2.0)
    axes[1].pcolormesh(t2, f2, zangle, vmin=0, vmax=2.0)
    axes[2].plot(t, x, to, xo)

    plt.figure(num=2)
    plt.loglog(freq, sp_mag)
    # freq, np.angle(sp))
    plt.xlabel('frequency')
    plt.ylabel('magnitude')

sind = 40000
find = sind + 1000
# print x[20000:20100]

if False:
    pya = pyaudio.PyAudio()
    stream = pya.open(format=pyaudio.paFloat32, channels=1, rate=fs, output=True)
    print 'play'
    # stream.write(xo)
    stream.write(x)
    print 'done'

    stream.stop_stream()
    stream.close()
    pya.terminate()
plt.show()
