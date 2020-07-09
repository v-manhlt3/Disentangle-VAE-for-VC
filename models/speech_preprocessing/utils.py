import numpy as np 
import librosa
import scipy

WINDOW_SIZE = 400
SHIFT_SIZE = 160

class SpeechSegment(Object):
    def __init__(self, signal):
        