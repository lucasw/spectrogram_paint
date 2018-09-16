# The timestamp ought to be of the first sample
# The frame ought to be where the in space the sample was recorded
# if from a microphone, or where a speaker is that is outputting the audio
std_msgs/Header header
# multichannel audio ought to have been recorded at different points in space
# (or the same but with different directional pose)
# so the way to accomplish stereo is multiples of this message
# either with a synchronizing filter or make a new message with an
# array of these messages within it.
float32[] data
float32 sample_rate
