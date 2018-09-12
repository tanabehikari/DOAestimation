import math
import numpy as np
#import ASP   as asp
import libs.stft
import libs.utils
#import stft
'''
sound source Segregation based on estimating incident Angle of
each Frequency component of Input signals Acquired by multiple 
microphones (SAFIA)
!!! Only the condition "2 source - 2 microphone" is supported !!!
--------
[Reference]
M. Aoki, M. Okamoto, S. Aoki, H. Matsui, T. Sakurai and Y. Kaneda,
"Sound source segregation based on estimating incident angle of 
each frequency component of input signals aquired by multiple 
microphones,"
Acoust. Sci. & Tech., vol. 22, no, 2, pp. 149--157, 2001.
--------
Created on 2018/04/06
@author: Naohiro TAWARA (KOBAYASHI Lab., Waseda Univ.)
(Original version was created by Motoi OMACHI on 2013/09/05
--------
'''

def generate_mask(x1, x2, feat_type, threshold):
    num_frames, hfftl = x1.shape
    mask = np.zeros((2, num_frames, hfftl), dtype=np.complex128)
    if feat_type is 'LR':
        p0  = np.abs(x1)
        p1  = np.abs(x2)
        mask[0] = (p0 - p1 > threshold)
        mask[1] = (p0 - p1 < threshold)
    elif feat_type is 'PD':
        f = np.angle((x1 / abs(x1)) / (x2 / abs(x2)))
        mask[0] = (abs(f) < threshold)
        mask[1] = (abs(f) > threshold)
    else:
        raise NameError('Invalid feat type: {}'.format(feat_type))
    '''
    elif feat_type is 'BF':
        x1_ = asp.BeamFormer(x1, x2, Channel=(0,1), Angle=90.0, Shape='Card')
        x2_ = asp.BeamFormer(x1, x2, Channel=(1,0), Angle=90.0, Shape='Card')
        p0 = np.abs(x1_)
        p1 = np.abs(x2_)
        mask[0] = (p0 - p1 > threshold)
        mask[1] = (p0 - p1 < threshold)
    '''
    return mask

def safia(x1, x2, frame_size=1024, frame_shift=256, fftl=1024, fs=16000, feat_type='PD', threshold=0.0, wav_out=True,input_is_wave = True):
    """
    Apply safia
    :param x1:          Input wave1.
    :param x2:          Input wave2.
    :param feat_type:   Type of masking
                        (1st microphone is assumed as a ba
                            se channel)
                            * 'LR': Level Ratio
                            * 'BF': Level Ratio (Beamformer output)
                            * 'PD': Phase difference
    :param threshold:   Threshold of masking
    :param wave_out:    If True, output waves
    :return:            Separated waves and mask
    """

    # Convert to spectrogram
    if input_is_wave:
        x1_, syn_param1 = libs.stft.stft(x1, frame_size, frame_shift, fftl, fs)
        x2_, syn_param2 = libs.stft.stft(x2, frame_size, frame_shift, fftl, fs)

    ### --- Mask Generation --- ###
    mask = generate_mask(x1_, x2_, feat_type, threshold)

    ### --- T-F Masking --- ###
    y1 = x1_ * mask[0]
    y2 = x1_ * mask[1]  #x2_??"
 
    ### --- Synthesize wave --- ###
    if wav_out:
        y1 = libs.stft.synth(y1, syn_param1)
        y2 = libs.stft.synth(y2, syn_param2)

    return y1, y2, abs(mask), syn_param1



def cardioidBF(x1,x2,MicInterbal_m=0.03,NullAngle=90,SoundVelocity=340,fs=16000,frame_size=1024, frame_shift=256,fftl=1024):
    # Convert to spectrogram
    x1_, syn_param1 = libs.stft.stft(x1, frame_size, frame_shift, fftl, fs)
    x2_, syn_param2 = libs.stft.stft(x2, frame_size, frame_shift, fftl, fs)

    # Beamforming
    hfftl1,nFlame1 = x1_.shape
    hfftl2,nFlame2 = x2_.shape
    freqvec = fs/((hfftl1-1)*2)* np.matrix([i for i in range(hfftl1)])   
    omega = 2*math.pi*freqvec.T
    tau = MicInterbal_m*math.sin(NullAngle)/SoundVelocity
    e = np.exp(-1j*omega*tau)*np.ones((1,nFlame1))
    e = np.array(e)

#    return x1_,x2_,omega,tau,e
    cardioidBeamformedWave = x1_ - x2_*e
    return  cardioidBeamformedWave,syn_param1

def minus(x1,x2,MicInterbal_m=0.03,NullAngle=90,SoundVelocity=340,fs=16000,frame_size=1024, frame_shift=256,fftl=1024):
    # Convert to spectrogram
    x1_, syn_param1 = libs.stft.stft(x1, frame_size, frame_shift, fftl, fs)
    x2_, syn_param2 = libs.stft.stft(x2, frame_size, frame_shift, fftl, fs)

    # Beamforming
    Y = x1_ - x2_
    y = libs.stft.synth(Y,syn_param1)
    return y,syn_param1