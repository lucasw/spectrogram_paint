#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
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
    time = np.arange(N) / float(fs)
    carrier = amp * np.sin(2*np.pi*50*time)
    noise = np.random.normal(scale=np.sqrt(noise_power),
                             size=time.shape)
    x = carrier + noise

    f, t, Zxx = signal.stft(x, fs=fs, nperseg=nperseg, return_onesided=onesided)
    zmag = np.abs(Zxx)
    zphase = np.angle(Zxx)
    print Zxx.shape, Zxx[1,1], zmag[1,1], zphase[1,1]
    print np.min(zmag), np.max(zmag)
    print np.min(zphase), np.max(zphase)

mag_image = sys.argv[1]
phase_image = sys.argv[2]
im = np.flipud(io.imread(mag_image))

print mag_image, im.shape, np.min(im), np.max(im)
if len(im.shape) > 2:
    print 'warning image has multiple layers', im.shape[2]
    im = im[:,:,0]
# print im

phase = io.imread(phase_image)
print phase_image, phase.shape, np.min(phase), np.max(phase)

zxx = im * np.exp(1j * phase)

fs = 44100
to, x_unfiltered = signal.istft(zxx, fs=fs, input_onesided=onesided)

# filter out the DC and the higher frequencies
xo = butter_bandpass_filter(x_unfiltered, lowcut=20.0, highcut=10000.0, fs=fs, order=3)

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
# plt.show()

sp = np.fft.fft(x)
half_ind = sp.shape[-1] / 2
sp_mag = np.abs(sp[:half_ind])
print 'fft min max', np.min(sp_mag), np.max(sp_mag)
freq = np.fft.fftfreq(t.shape[-1], 1.0 / fs)
freq = freq[:half_ind]



if False:
    fig, axes = plt.subplots(3, 1, num=1) # sharedy=True) # figure(num=0)
    # print axes.shape
    axes[0].pcolormesh(t2, f2, logzmag, vmin=0, vmax=2.0)
    axes[1].pcolormesh(t2, f2, zangle, vmin=0, vmax=2.0)
    axes[2].plot(t, x, to, xo)

    plt.figure(num=2)
    plt.loglog(freq, sp_mag)
    # freq, np.angle(sp))
    plt.xlabel('frequency')
    plt.ylabel('magnitude')

x = x / np.max(np.abs(x))
wavfile.write("test.wav", fs, x)

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
