import numpy as np
import copy as cp
import libs.utils as u

"""
Short-time Fourier Transform
--------
Created on 2018/04/05
@author: Naohiro TAWARA (KOBAYASHI Lab., Waseda Univ.)
--------
"""

def stft(wave, frame_size=1024, frame_shift=256, fftl=1024, fs=16000, window_type='hamming'):
    """
    Short-time Fourier transform
    :param wave:        Raw input
    :param frame_size:  Frame size
    :param frame_shift: Frame shift
    :param fftl:        Number of bins
    :param fs:          Sampling frequency
    :param window_type: Window type
    :return:            Spectrogram and SynParam
    """
    try:
        # Multiple channels
        num_mics, num_samples = wave.shape
    except:
        # Single channel
        num_mics = 1
        num_samples = len(wave)

    hfftl = int(np.ceil(fftl / 2 ))
    num_frames = int(np.floor((num_samples - frame_size) / frame_shift) + 1)
    window = u.window1d(frame_size, window_type)

    if num_mics > 1:
        X = np.zeros((num_mics, num_frames, hfftl ), dtype = np.complex128)
    else:
        X = np.zeros((num_frames, hfftl ), dtype = np.complex128)

    for mic in range(num_mics):
        for n in range(num_frames):
            #% --- frame numbers --- %
            bf = n * frame_shift
            ef = bf + frame_size
            #% --- Spectrum --- %
            if num_mics > 1:
                x_n = wave[mic, bf: ef] * window
                X[mic, n] = np.fft.fft(x_n, fftl)[0 : hfftl ]
            else:
                x_n = wave[bf: ef] * window
                X[n] = np.fft.fft(x_n, fftl)[0 : hfftl ]

    return X, SynParam(num_samples, num_frames, frame_size, frame_shift, fftl, fs, window_type)

def synth(x, syn_param, phase = None, low_freq=300, up_freq=5499, is_bpf=True):
    """
    Generate waveform from Spectrogram
    :param x:           Spectrogram
    :param syn_param    Paramter of short time Fourier transform
    :param low_freq:    lower frequency
    :param up_freq:     upper frequency
    :param is_bpf:      If True, band pass filter will be applied
    :return:            Wave
    """
    assert syn_param.num_frames == x.shape[0], \
        "Number of frames is mismatched: {} != {}".format(syn_param.num_frames, x.shape[0])
    if is_bpf:
        bpf_filter = bpf(int(np.ceil(syn_param.fftl / 2 )), syn_param.fs, low_freq, up_freq)
        bpf_filter = np.append(bpf_filter, bpf_filter[::-1] * 0)
    num_samples = syn_param.num_samples
    frame_size = syn_param.frame_size
    frame_shift = syn_param.frame_shift
    window = u.window1d(frame_size, syn_param.window_type)
    fftl = syn_param.fftl
    num_frames = x.shape[0]
    hfftl = x.shape[1]
    y = np.zeros(num_samples)
    for n in range(num_frames):
        x_t = x[n]
        x_t_reverse = np.conj(x[n][-1 : - hfftl - 1 : -1])
        x_ = np.append(x_t, x_t_reverse)
        if phase is not None:
            phase_t = phase[n]
            phase_t_reverse = np.conj(phase[n][-1: - hfftl - 1: -1])
            phase_ = np.append(phase_t, phase_t_reverse)
            x_ = x_ * np.exp(1j * phase_)
        #% --- frame numbers --- %
        bf = n * frame_shift
        ef = bf + frame_size
        #% --- Overlap and Add --- %
        if is_bpf:
            y_ = np.fft.ifft(x_ * bpf_filter, fftl)
        else:
            y_ = np.fft.ifft(x_, fftl)
        y_ = np.real(y_[: frame_size])
        y[bf : ef] += y_ * window
        # wsum[bf:ef] += WINDOW ** 2
    # pos = (wsum != 0)
    # y[pos] /= wsum[pos]
    return y

def bpf(hfftl, fs, low_freq, up_freq):
    fil = np.zeros(hfftl)
    num_low_freq = low_freq / float(fs) * (hfftl * 2.)
    num_up_freq = up_freq / float(fs) * (hfftl * 2.)
    fil[int(np.floor(num_low_freq)):int(np.floor(num_up_freq))] = 1.
    if int(np.floor(num_low_freq)) == int(np.ceil(num_low_freq)):
        fil[int(np.ceil(num_low_freq))] = 0.5
    else:
        fil[int(np.floor(num_low_freq))] = 1 / 3
        fil[int(np.ceil(num_low_freq))]  = 2 / 3
    if int(np.floor(num_up_freq)) == int(np.ceil(num_up_freq)):
        fil[int(np.ceil(num_up_freq))] = 0.5
    else:
        fil[int(np.floor(num_up_freq))] = 1 / 3
        fil[int(np.ceil(num_up_freq))]  = 2 / 3
    return fil

class SynParam:
    def __init__(self, num_samples, num_frames, frame_size, frame_shift, fftl, fs, window_type):
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.fftl = fftl
        self.fs = fs
        self.window_type = window_type


class Spectrogram:
    """
    Spectrogam    
    --------
    Created on 2018/04/05
    @author: Naohiro TAWARA(KOBAYASHI Lab., Waseda Univ.)
    --------
    """
    
    def __init__(self, X, num_samples, num_frames, frame_size, frame_shift, fftl, fs, window_type):
        """
        Initialization of Spec class
        :param FrameSize:   Frame size
        :param FrameShift:  Frame shift
        :param FFTL:        Number of bins
        :param Fs:          Sampling frequency
        :param WindowType:  Type of analysis window
        """

        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.fftl = fftl
        self.fs = fs
        self.window_type = window_type
        self.data = X
        self.num_samples = num_samples
        self.num_frames = num_frames

    def show_info(self, Detail=False):
        """
        Display Information about Local Feature Extraction
        <<Input>>
        Detail  ... Display parameter in detail
        <<Output>>
        None
        """
        print("================ Spec =================")
        print("FrameSize   : {}".format(str(self.frame_size)))
        print("FrameShift  : {}".format(str(self.frame_shift)))
        print("FFTL        : {}".format(str(self.fftl)))
        print("Numfrms     : {}".format(str(self.num_frames)))
        print("Fs          : {}".format(str(self.fs)))
        print("WindowType  : {}".format(self.window_type))
        print("Format      : {}").format(self.format)
        print("Component   : [{}, {}]".format(str(self.data.shape), str(self.data.dtype)))
        if Detail:
            print(self.data)
        print("=" * 40)
    
    def get(self, form = 'Complex'):
        """
        Get value.
        :param form: Spectrgoram format
                      * 'Complex'   : Complex Spectrum
                      * 'Amplitude' : Amplitude Spectrum
                      * 'LPSpec'    : Log-Power Spectrum
                      * 'Phase'     : Phase Spectrum
        :return:     Local feature
        """
        x = self.data.copy()
        if form is 'Complex':
            pass
        elif form is 'Amplitude':
            x = abs(x)
        elif form is 'LPSpec':
            x = abs(x)
            x[x < u.MinV] = u.MinV
            x = 20.0 * np.log10(x)
        elif form is 'Phase':
            x = x / np.abs(x)
        else:
            raise('Format error exception')
        
        return x

    def copy(self):
        return cp.copy(self)
