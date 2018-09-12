# import scipy as sp
import numpy as np
# import copy as cp
# import Util as u
# import STFT as s
# import matplotlib.pyplot as plt

'''
Speech Signal Processing
--------
Created on 2014/02/13
@author: Motoi OMACHI(KOBAYASHI Lab., Waseda Univ.)
--------
'''

def spec2cep(x, ord=20, eps=1e-10):
    """
    Extract cepstrum coefficient from Spectrum
    :param x:   Spectrum
    :param ord: Order of cepstrum coefficient
    :param eps: Flooring value
    :return:    Cepstrum
    """
    x = np.hstack((x, np.fliplr(x[:,1:-1])))
    abs_x = abs(x)
    abs_x[abs_x < eps] = eps
    c = np.real(np.fft.ifft(np.log(abs_x)))
    c = c[:, 0: ord]
    
    return c

def cep2spec(c, FFTL):
    """
    Reconst Spectrum from cepstrum coefficient
    :param c:       Cepstrum coefficient
    :param FFTL:    FFT length
    :return:        Spectrum
    """
    NFrame,ord = c.shape
    C = np.zeros((NFrame, FFTL),dtype = np.float64)
    logX = np.zeros((NFrame, FFTL), dtype = np.complex64)
    C[:, 0:ord] = c
    C[:, (FFTL-ord+1):] = np.fliplr(c[:, 1:])
    for frm in range(NFrame):
        logX[frm, :] = np.fft.fft(C[frm, :], FFTL)
    X = np.exp(abs(logX))

    return X[:, 0:FFTL / 2 + 1]

def calcSPD(X, X_tar, eps = 1e-10, PNorm = False, ALLFrq = False, Segmental = False):
    """
    Calculate Correlation of Spectrum1 and Spectrum2
    :param X:           Spectrum 1
    :param X_tar:       Spectrum 2
    :param eps:         Flooring value
    :param PNorm:       Power normalization
    :param ALLFrq:
    :param Segmental:
    :return:            Spectral distance
    """

    ### --- Settings --- ###
    NFrame_ref, HFFTL = X.shape
    NFrame_tar, HFFTL = X_tar.shape
    NFrame = np.min([NFrame_ref, NFrame_tar])
    X     = X[:NFrame]
    X_tar = X_tar[:NFrame]

    ### --- Log Power Spec --- ###
    X = abs(X)
    X[X < eps] = eps
    X_tar = abs(X_tar)
    X_tar[X_tar < eps] = eps
    X     = 20.0 * np.log10(X)
    X_tar = 20.0 * np.log10(X_tar)

    P     = np.mean(X_tar, axis = 1)
    V_seg = np.argwhere(P > np.mean(P[:3]))
    #if PNorm:
    if Segmental:
        print("----")
        print((np.mean(X[V_seg,:],axis=1)).shape)
        print(V_seg.shape)
        X_       = X[V_seg,:]-np.mean(X[V_seg,:],axis=1)
        X_tar_   = X_tar[V_seg,:]-np.mean(X_tar[V_seg,:],axis=1)
        #X        = X_
        #X_tar    = X_tar_
        # else:
        X       = X-np.mean(X,axis=1).reshape(NFrame,1)
        X_tar   = X_tar-np.mean(X_tar,axis=1).reshape(NFrame,1)
    if ALLFrq:
        D = np.mean(np.sqrt((X_tar[0 : NFrame, :] - X[0:NFrame, :]) ** 2), axis=0)
    else:
        D = np.mean(np.sqrt(np.sum((X_tar[0 : NFrame, :] - X[0 : NFrame, :]) ** 2, axis=1) / HFFTL))
    
    return D

def calculating_correlation(x, x_tar, eps=1e-10, pnorm=False, all_freq=True):
    """
    Calculate Spectral Distance between Spectrums
    :param x:       Input spectrum1
    :param x_tar:   Input spectrum2
    :param eps:     Flooring value
    :param pnorm:   Power normalization
    :param all_freq:
    :return:        Spectrum distance
    """
    ####################
    ### --- Main --- ###
    ####################
    ### --- Settings --- ###
    num_frames_ref, hfftl = x.shape
    num_frames_tar, hfftl = x_tar.shape
    num_frames = np.min([num_frames_ref, num_frames_tar])

    ### --- Log Power Spec --- ###
    x = abs(x)
    x[x < eps] = eps
    x_tar = abs(x_tar)
    x_tar[x_tar < eps] = eps
    x = 20.0 * np.log10(x)
    x_tar = 20.0 * np.log10(x_tar)
    if pnorm:
        x = x - np.mean(x, axis=1).reshape(num_frames, 1)
        x_tar = x_tar - np.mean(x_tar, axis=1).reshape(num_frames, 1)
    
    dist = np.zeros((1, hfftl), dtype=np.float32)
    if all_freq:
        for frq in range(hfftl):
            xf = x[:, frq]
            yf = x_tar[:,frq]
            x_mu  = np.mean(xf)
            x_ = xf - np.dot(x_mu, np.ones((1, num_frames_ref)))
            y_mu = np.mean(yf)
            y_ = yf - np.dot(y_mu, np.ones((1, num_frames_tar)))
            dist[0, frq] = np.sum(x_ * y_) / (np.sqrt(np.sum(x_ * x_)) * np.sqrt(np.sum(y_ * y_)))
    else:
        pass
    
    return dist



# if __name__ == "__main__" :
#     ### --- Analysis --- ###
#     Fs=16000.0
#     FrameSize=512
#     FrameShift=np.ceil(0.005*Fs)
#     FFTL=1024
#     WindowType='Hanning'
#     S=s.Spec(FrameSize,FrameShift,FFTL,Fs,WindowType)
#     
#     x=u.RawRead('a001.pcm')
#     s.STFT(x,S)
#     X_env=S.Get()
#     c=Spec2Cep(X_env,20)
#     ReconstX=Cep2Spec(c,FFTL)
#     
#     plt.figure()
#     plt.plot(20.0*np.log10(abs(X_env[100,:])))
#     plt.hold(True)
#     plt.plot(20.0*np.log10(abs(ReconstX[100,:])),'r')
#     plt.show()
#     
=======
# import scipy as sp
import numpy as np
# import copy as cp
# import Util as u
# import STFT as s
# import matplotlib.pyplot as plt

'''
Speech Signal Processing
--------
Created on 2014/02/13
@author: Motoi OMACHI(KOBAYASHI Lab., Waseda Univ.)
--------
'''

def spec2cep(x, ord=20, eps=1e-10):
    """
    Extract cepstrum coefficient from Spectrum
    :param x:   Spectrum
    :param ord: Order of cepstrum coefficient
    :param eps: Flooring value
    :return:    Cepstrum
    """
    x = np.hstack((x, np.fliplr(x[:,1:-1])))
    abs_x = abs(x)
    abs_x[abs_x < eps] = eps
    c = np.real(np.fft.ifft(np.log(abs_x)))
    c = c[:, 0: ord]
    
    return c

def cep2spec(c, FFTL):
    """
    Reconst Spectrum from cepstrum coefficient
    :param c:       Cepstrum coefficient
    :param FFTL:    FFT length
    :return:        Spectrum
    """
    NFrame,ord = c.shape
    C = np.zeros((NFrame, FFTL),dtype = np.float64)
    logX = np.zeros((NFrame, FFTL), dtype = np.complex64)
    C[:, 0:ord] = c
    C[:, (FFTL-ord+1):] = np.fliplr(c[:, 1:])
    for frm in range(NFrame):
        logX[frm, :] = np.fft.fft(C[frm, :], FFTL)
    X = np.exp(abs(logX))

    return X[:, 0:FFTL / 2 + 1]

def calcSPD(X, X_tar, eps = 1e-10, PNorm = False, ALLFrq = False, Segmental = False):
    """
    Calculate Correlation of Spectrum1 and Spectrum2
    :param X:           Spectrum 1
    :param X_tar:       Spectrum 2
    :param eps:         Flooring value
    :param PNorm:       Power normalization
    :param ALLFrq:
    :param Segmental:
    :return:            Spectral distance
    """

    ### --- Settings --- ###
    NFrame_ref, HFFTL = X.shape
    NFrame_tar, HFFTL = X_tar.shape
    NFrame = np.min([NFrame_ref, NFrame_tar])
    X     = X[:NFrame]
    X_tar = X_tar[:NFrame]

    ### --- Log Power Spec --- ###
    X = abs(X)
    X[X < eps] = eps
    X_tar = abs(X_tar)
    X_tar[X_tar < eps] = eps
    X     = 20.0 * np.log10(X)
    X_tar = 20.0 * np.log10(X_tar)

    P     = np.mean(X_tar, axis = 1)
    V_seg = np.argwhere(P > np.mean(P[:3]))
    #if PNorm:
    if Segmental:
        print("----")
        print((np.mean(X[V_seg,:],axis=1)).shape)
        print(V_seg.shape)
        X_       = X[V_seg,:]-np.mean(X[V_seg,:],axis=1)
        X_tar_   = X_tar[V_seg,:]-np.mean(X_tar[V_seg,:],axis=1)
        #X        = X_
        #X_tar    = X_tar_
        # else:
        X       = X-np.mean(X,axis=1).reshape(NFrame,1)
        X_tar   = X_tar-np.mean(X_tar,axis=1).reshape(NFrame,1)
    if ALLFrq:
        D = np.mean(np.sqrt((X_tar[0 : NFrame, :] - X[0:NFrame, :]) ** 2), axis=0)
    else:
        D = np.mean(np.sqrt(np.sum((X_tar[0 : NFrame, :] - X[0 : NFrame, :]) ** 2, axis=1) / HFFTL))
    
    return D

def calculating_correlation(x, x_tar, eps=1e-10, pnorm=False, all_freq=True):
    """
    Calculate Spectral Distance between Spectrums
    :param x:       Input spectrum1
    :param x_tar:   Input spectrum2
    :param eps:     Flooring value
    :param pnorm:   Power normalization
    :param all_freq:
    :return:        Spectrum distance
    """
    ####################
    ### --- Main --- ###
    ####################
    ### --- Settings --- ###
    num_frames_ref, hfftl = x.shape
    num_frames_tar, hfftl = x_tar.shape
    num_frames = np.min([num_frames_ref, num_frames_tar])

    ### --- Log Power Spec --- ###
    x = abs(x)
    x[x < eps] = eps
    x_tar = abs(x_tar)
    x_tar[x_tar < eps] = eps
    x = 20.0 * np.log10(x)
    x_tar = 20.0 * np.log10(x_tar)
    if pnorm:
        x = x - np.mean(x, axis=1).reshape(num_frames, 1)
        x_tar = x_tar - np.mean(x_tar, axis=1).reshape(num_frames, 1)
    
    dist = np.zeros((1, hfftl), dtype=np.float32)
    if all_freq:
        for frq in range(hfftl):
            xf = x[:, frq]
            yf = x_tar[:,frq]
            x_mu  = np.mean(xf)
            x_ = xf - np.dot(x_mu, np.ones((1, num_frames_ref)))
            y_mu = np.mean(yf)
            y_ = yf - np.dot(y_mu, np.ones((1, num_frames_tar)))
            dist[0, frq] = np.sum(x_ * y_) / (np.sqrt(np.sum(x_ * x_)) * np.sqrt(np.sum(y_ * y_)))
    else:
        pass
    
    return dist



# if __name__ == "__main__" :
#     ### --- Analysis --- ###
#     Fs=16000.0
#     FrameSize=512
#     FrameShift=np.ceil(0.005*Fs)
#     FFTL=1024
#     WindowType='Hanning'
#     S=s.Spec(FrameSize,FrameShift,FFTL,Fs,WindowType)
#     
#     x=u.RawRead('a001.pcm')
#     s.STFT(x,S)
#     X_env=S.Get()
#     c=Spec2Cep(X_env,20)
#     ReconstX=Cep2Spec(c,FFTL)
#     
#     plt.figure()
#     plt.plot(20.0*np.log10(abs(X_env[100,:])))
#     plt.hold(True)
#     plt.plot(20.0*np.log10(abs(ReconstX[100,:])),'r')
#     plt.show()
#     
>>>>>>> d8c5583da6d9f9781cca014d1f3fc63614f80e5f
