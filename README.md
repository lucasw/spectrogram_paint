# spectrogram_paint
Paint a floating point image and use it as a spectrogram to update audio output live

## Dependencies

Need python skimage

sudo apt install python-skimage


Create 32-bit float images with Gimp 2.9+ (actually any images probably work)

./load_32bit_tif.py ~/Documents/floatimage.tif ~/Documents/phaseimage.tif

The images need to be the same size.
