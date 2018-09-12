
import numpy as np
from scipy import  zeros
###import scipy.fftpack as fft

class SynParam():
    def __init__(self, Fs, N_samples, FRAME_SIZE, FRAME_SHIFT, N_FRAMES, FFTL, WINDOW):
        self.Fs = Fs
        self.N_samples = N_samples
        self.FRAME_SIZE = FRAME_SIZE
        self.FRAME_SHIFT = FRAME_SHIFT
        self.N_FRAMES = N_FRAMES
        self.FFTL = FFTL
        self.WINDOW = WINDOW

def FFTanalysis(M, FRAME_SIZE = 1024, FRAME_SHIFT = 256, FFTL = 1024, Fs = 16000, WindowType = 'hamming'):
    # M = one mic signal sequence
    # --- spectrum analysis ---
    HFFTL = int(FFTL / 2)
    N_samples = len(M)
    N_FRAMES = int(np.floor((N_samples - FRAME_SIZE) /FRAME_SHIFT) + 1)
    if WindowType == 'hamming':
        WINDOW = np.hamming(FRAME_SIZE)
    elif WindowType == 'hanning':
        WINDOW = np.hanning(FRAME_SIZE)
    else:
        WINDOW = np.hamming(FRAME_SIZE)

    X = zeros((N_FRAMES, HFFTL + 1), complex)
    for n in range(N_FRAMES):
        #% --- frame numbers --- %
        bf = n * FRAME_SHIFT
        ef = bf + FRAME_SIZE
        # print "{} {}".format(bf,ef)
        #% --- Spectrum --- %
        x_n = M[bf:ef] * WINDOW
        X[n] = np.fft.fft(x_n, n = FFTL)[0:HFFTL + 1]
    SynP = SynParam(Fs, N_samples, FRAME_SIZE, FRAME_SHIFT, N_FRAMES, FFTL, WINDOW)
    return (X, SynP)


def Synth(X, SynP, LowFreq = 300, UpFreq = 5499, BPFon = 1):
    if BPFon == 1:
        BPF = bpf(SynP.FFTL / 2, SynP.Fs, LowFreq, UpFreq)
        BPF = np.append(BPF, BPF[::-1] * 0)
    N_samples = SynP.N_samples
    FRAME_SIZE = SynP.FRAME_SIZE
    FRAME_SHIFT = SynP.FRAME_SHIFT
    WINDOW = SynP.WINDOW
    FFTL = SynP.FFTL
    N_FRAMES = np.shape(X)[0]
    HFFTL = np.shape(X)[1]
    y = np.zeros(N_samples)
    for n in range(N_FRAMES):
        X_t = X[n]
        X_t_reverse = np.conj(X[n][-2:-HFFTL:-1])
        X_ = np.append(X_t, X_t_reverse)
        #% --- frame numbers --- %
        bf = n * FRAME_SHIFT
        ef = bf + FRAME_SIZE
        #% --- Overlap and Add --- %
        if BPFon == 1:
            y_ = np.fft.ifft(X_ * BPF, FFTL)
        else:
            y_ = np.fft.ifft(X_, FFTL)
        y_ = np.real(y_[:FRAME_SIZE])
        y[bf:ef] += y_ * WINDOW
        # wsum[bf:ef] += WINDOW ** 2
    # pos = (wsum != 0)
    # y[pos] /= wsum[pos]
    return y

def bpf(HFFTL, Fs, LowFreq, UpFreq):
    fil = np.zeros(HFFTL)
    N_LowFreq = LowFreq / Fs * (HFFTL * 2.)
    N_UpFreq = UpFreq / Fs * (HFFTL * 2.)
    fil[int(np.floor(N_LowFreq)):int(np.floor(N_UpFreq))] = 1.
    if int(np.floor(N_LowFreq)) == int(np.ceil(N_LowFreq)):
        fil[int(np.ceil(N_LowFreq))] = 0.5
    else:
        fil[int(np.floor(N_LowFreq))] = 1 / 3
        fil[int(np.ceil(N_LowFreq))]  = 2 / 3
    if int(np.floor(N_UpFreq)) == int(np.ceil(N_UpFreq)):
        fil[int(np.ceil(N_UpFreq))] = 0.5
    else:
        fil[int(np.floor(N_UpFreq))] = 1 / 3
        fil[int(np.ceil(N_UpFreq))]  = 2 / 3
    return fil
