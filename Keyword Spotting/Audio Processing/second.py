import tensorflow as tf
import numpy as np

numcep = 390
with tf.compat.v1.Session() as sess:
    filename = '123346__matteusnova__hello.wav'
    raw_audio = tf.io.read_file(filename)
    audio, fs = tf.audio.decode_wav(raw_audio)
    spectrogram = tf.audio.compat.v1.audio_ops.audio_sprectrogram(audio, window_size = 124, stride = 64)
    orig_inputs = tf.audio.compat.v1.audio_ops.mfcc(spectrogram, sample_rate = fs, dct_coefficient_count=numcep)

    audio_mfcc = orig_inputs.eval()
    print(np.shape(audio_mfcc))
