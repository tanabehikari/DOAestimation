
import scipy.signal
from scipy.io.wavfile import read,write
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
#import libs.FFT as FFT
import os
import numpy as np
import random
import string
MinV=1e-10

import logging

logger = logging.getLogger('__name__')
ch=logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)


def plot_wave(wav, fs, output_filename=None, title=None):
    plt.figure()
    plt.plot([x / fs * 1000 for x in range(len(wav))], wav)
    plt.xlabel('time [ms]')
    plt.ylabel('amplitude')
    if title is not None:
        plt.title(title)
    if output_filename is not None:
        plt.savefig(output_filename)

def power_normalize(input, target_norm):
    #return input / (target_norm)
    coefficient = np.sqrt(target_norm / np.mean(input**2))
    return coefficient

def down_sampling(wav, fs, Fs):
    return scipy.signal.decimate(wav, int(np.ceil(fs / Fs)), n  = 512, ftype = "fir", zero_phase = True)

def read_int16_as_float32(filename, Fs=None):
    try:
        fs, wav = read(filename)
    except scipy.io.wavfile.WavFileWarning:
        pass

    wav = wav / 32768.
    if Fs is not None and fs != Fs:
        wav = down_sampling(wav, fs, Fs)
    return wav

def write__float32_as_int16(filename, fs, data):
    #assert abs(data.max())<=1.0, "Clipping is needed for {}: {}".format(filename, abs(data.max()))

    if abs(data.max()) > 1.0:
        print("Clipping is needed for {}: {}".format(filename, abs(data.max())))
        data[data>1.0] = 1.0
        data[data < -1.0] = -1.0
    write(filename, fs, np.asarray(data * 32768., dtype = np.int16))

def load_impulse(impfile, fs = None, imp_length = 1024, holding = 20):
    imp = read_int16_as_float32(impfile, fs)
    peak = np.argmax(imp)
    imp = imp[range(peak - holding, peak - holding + imp_length)]
    #H, params = FFT.FFTanalysis(imp)
    return imp

def get_output_dirname(ch, noise_id, snr):
    return os.path.join("ch{}".format(ch + 1), "{}dB".format(snr), "noise{}".format(noise_id))

def make_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def apply_convolution(voice, H):
    X, synparamX = FFT.FFTanalysis(voice)
    X = X * H #impulse response convolution
    Y = FFT.Synth(X, synparamX, BPFon = 0)
    return Y


def calcurate_SNR(voice, noise):
    assert len(voice) == len(noise), "Duration mismatch (voice:{}, noise:{})".format(len(voice), len(noise))
    return 20 * np.log10(np.linalg.norm(voice - noise) / np.linalg.norm(noise))


def NormNormalize(x,RemoveDC=False):
    '''
    Norm Normalization
    <<Input>>
    x    ... Data
    RemoveDC ... Remove DC Component
    <<Output>>
    y    ... Normalized data
    Norm ... Norm
    '''
    Dim,NSample=x.shape

    if RemoveDC:
        x_mu=np.mean(x,axis=1).reshape(Dim,1)
        x=x-x_mu

    x_norm=np.sqrt(np.dot(np.ones((1,Dim),dtype=np.float64),abs(x)**2))
    x_norm[np.isnan(x_norm)]=MinV
    x_norm[x_norm<MinV]=MinV
    y=x/np.dot(np.ones((Dim,1),dtype=np.float64),x_norm)

    return y,x_norm

def chunked(iterable, n):
    ''' Seperate iterable set into subset which contain n elements'''
    return [iterable[x:x + n] for x in range(0, len(iterable), n)]


def _CalcZC(x, FSize=1024, FShift=256):
    '''
    Count Zero-cross num.
    <<Input>>
    x      ... Waveform
    FSize  ... Frame size
    FShift ... Frame shift
    <<Output>>
    ZC     ... The num. of zero-cross
    '''
    ##################################
    ### --- Parameter Settings --- ###
    ##################################
    x = np.float64(x)
    NSample = len(x)
    NFrames = int(np.floor((NSample - FSize) / FShift) + 1)

    ####################
    ### --- Main --- ###
    ####################
    ### --- Memory allocation --- ###
    ZC = np.zeros((NFrames, 1), dtype=np.float64)
    E = np.zeros((NFrames, 1), dtype=np.float64)
    ### --- Frame-wise processing --- ###
    bfrm = 0
    for nfrm in range(NFrames):
        if bfrm + FSize > x.size:
            x_frm_ = x[bfrm:x.size]
            ZC[nfrm] = len(np.argwhere(x_frm[:-1] * x_frm[1:] < 0))
        else:
            x_frm = x[bfrm:bfrm + FSize]
            ZC[nfrm] = len(np.argwhere(x_frm[:-1] * x_frm[1:] < 0))
            bfrm += FShift
        E[nfrm] = np.average(np.abs(x_frm))

    return ZC, E


def VAD(x, thre=0.0001, FSize=1024, FShift=256):
    _, E = _CalcZC(x)
    NSample = len(x)
    vad = np.zeros(len(x), dtype=np.int32)
    NFrames = int(np.floor((NSample - FSize) / FShift) + 1)
    for nfrm in range(NFrames):
        vad[nfrm * FShift:(nfrm + 1) * FShift] = [E[nfrm, 0] > thre] * FShift
    return vad

def get_random_str(n=5):
    return ''.join([random.choice(string.ascii_letters + string.digits) for i in range(n)])


def window1d(window_size=10, window_type='hanning'):
    """
    Window functions
    :param window_size: Window size
    :param window_type: Window type
                          * 'Hanning'   : Hanning Window
                          * 'Hanning2'  : Hanning Window (pi delayed)
                          * 'Rectangle' : Tectangular window
    :return:            Window function
    """
    if window_type is 'hanning':
        t = np.array(range(0, window_size), dtype = np.float64)
        window = 0.5 - 0.5 * np.cos(2 * np.pi * t / window_size)
    elif window_type is 'hanning2':
        t = np.array(range(0, window_size),dtype = np.float64)
        window = 0.5 - 0.5 * np.cos(2 * np.pi * t / window_size + np.pi)
    elif window_type is 'hamming':
        t = np.array(range(0, window_size), dtype = np.float64)
        window = 0.54 - 0.46 * np.cos(2 * np.pi * t / window_size)
    elif window_type is 'rectangle':
        window = np.ones(window_size)
    else:
        logger.warning("{} is not supported window type.\nHanning window is used.".format(window_type))
        window = window1d(window_size)
    return window

'''
def window2d(y_size=10, x_size=10, window_position=''):
    """
    2D-Window function
    <<Input>>
    YSize       ... Window Size for Y-axis
    XSize       ... Window Size for X-axis
    WindowPos   ... Position of Window
    <<Output>>
    window      ... Window
    """
    windowX = window1d(x_size, 'Hanning')
    windowY = window1d(y_size, 'Hanning')
    if window_position == 'LT':
        windowX[0:np.ceil(XSize / 2)] = 1
        windowY[0:np.ceil(YSize / 2)] = 1
    elif window_position == 'T':
        windowY[0:np.ceil(YSize / 2)] = 1
    elif WindowPos == 'RT':
        windowY[0:np.ceil(YSize / 2)] = 1
        windowX[np.ceil(XSize / 2):] = 1
    elif WindowPos == 'R':
        windowX[np.ceil(XSize / 2):] = 1
    elif WindowPos == 'RD':
        windowX[np.ceil(XSize / 2):] = 1
        windowY[np.ceil(YSize / 2):] = 1
    elif WindowPos == 'D':
        windowY[np.ceil(YSize / 2):] = 1
    elif WindowPos == 'LD':
        windowY[np.ceil(YSize / 2):] = 1
        windowX[0:np.ceil(XSize / 2)] = 1
    elif WindowPos == 'L':
        windowX[0:np.ceil(XSize / 2)] = 1
    elif WindowPos == 'LT2':
        windowX[0:XSize] = 1
        windowY[0:np.ceil(YSize / 2)] = 1
    elif WindowPos == 'L2':
        windowX[0:XSize] = 1
    elif WindowPos == 'LD2':
        windowY[np.ceil(YSize / 2):] = 1
        windowX[0:XSize] = 1
    #     if WindowPos=='L':
    #         windowX[0:np.ceil(XSize)/2]=1
    #     elif WindowPos=='R':
    #         windowX[np.ceil(XSize)/2:]=1
    #     elif WindowPos=='T':
    #         windowY[0:np.ceil(YSize)/2]=1
    #     elif WindowPos=='D':
    #         windowY[np.ceil(YSize)/2:]=1
    #     elif WindowPos=='LT':
    #         windowX[0:np.ceil(XSize)/2]=1
    #         windowY[0:np.ceil(YSize)/2]=1
    #     elif WindowPos=='RT':
    #         windowX[np.ceil(XSize)/2:]=1
    #         windowY[0:np.ceil(YSize)/2]=1
    #     elif WindowPos=='LD':
    #        \ windowX[0:np.ceil(XSize)/2]=1
    #         windowY[np.ceil(YSize)/2:]=1
    #     elif WindowPos=='RD':
    #         windowX[np.ceil(XSize)/2:]=1
    #         windowY[np.ceil(YSize)/2:]=1
    #     else:
    #        pass
    Wx = np.zeros((YSize, XSize), dtype=np.float64)
    for ix in range(y_size):
        Wx[ix] = windowX
    Wy = sp.zeros((XSize, YSize), dtype=np.float64)
    for iy in range(XSize):
        Wy[iy] = windowY
    window = Wx * Wy.T
    return window
'''

def replace_min_v(M, MinV):
    '''
    Replace elements which is less than minimum value
    Note that More than 2 dimension is not supported
    <<Input>>
    M        ... nd array
    MinV     ... Minimum value
    <<Output>>
    M        ... Converted array
    '''
    try:
        # Matrix
        YSize, XSize = M.shape
        assert not isinstance(XSize, tuple), "Error(LocalFeqtureExt) : Unsupported array is input"
    except:
        # Vector
        XSize = M.shape
        XSize = XSize[0]
        YSize = 1

    for iy in range(YSize):
        if YSize > 1:
            M_ = M[iy]
        else:
            M_ = M
        # Comparison
        for ix in range(XSize):
            if M_[ix] <= MinV:
                M_[ix] = MinV
            else:
                pass
        if YSize > 1:
            M[iy] = M_
        else:
            M = M_
    return M


def replace_max_v(M, MaxV):
    '''
    Replace elements which is more than maximum value
    Note that More than 2 dimension is not supported
    <<Input>>
    M        ... nd array
    MaxV     ... Maximum value
    <<Output>>
    M        ... Converted array
    '''
    try:
        # Matrix
        YSize, XSize = M.shape
        assert not isinstance(XSize, tuple), "Error(LocalFeqtureExt) : Unsupported array is input"
    except:
        # Vector
        XSize = M.shape
        XSize = XSize[0]
        YSize = 1

    for iy in range(YSize):
        if YSize > 1:
            M_ = M[iy]
        else:
            M_ = M
        # Comparison
        for ix in range(XSize):
            if M_[ix] >= MaxV:
                M_[ix] = MaxV
            else:
                pass
        if YSize > 1:
            M[iy] = M_
        else:
            M = M_
    return M

