
import numpy as np
import scipy as sp
import copy  as cp
import Util  as u
import STFT  as stft

'''
Utility functions for Acoustic Signal Processing
--------
Created on 2013/09/06
@author: Motoi OMACHI(KOBAYASHI Lab., Waseda Univ.)
--------
'''
MinV = 1e-10

def beam_former(MArray, channels = (0,1), angle=90, shape='Card'):
    '''
    Cardioid-shaped Beamformer
    <<Input>>
    MArray      ... Microphone Array(BSS Format)
    Ch          ... Channel
    Angle       ... Null Angle [deg]
    Shape       ... Beam shape
                    * 'Card'   : Cardioid shaped
                    * 'Dipole' : Dipole shaped
    <<Output>>
    OutBF       ... Beamformer output
    '''
    ##################################
    ### --- Parameter Settings --- ###
    ##################################
    if MArray.Format is 'BSS':
        MArray.ConvForm('AS')
    else:
        pass
    
    tau = calc_delay_time(MArray, angle)
    
    ####################
    ### --- Main --- ###
    ####################
    if shape is 'Card':
        add_delay(MArray, tau, channels[0])
    elif shape is 'Dipole':
        pass
    else:
        print("Warning(ASP.BeamFormer) : Beam shape '{}' is not defined.".format(str(shape)))
        add_delay(MArray, tau, channels[0])
    
    X = MArray.get()
    
    Y = X[channels[1]] - X[channels[0]]
#     (Nmic,HFFTL,Nfrm)=X.shape
    
    return Y

def add_delay(MArray, tau = 0.0, channel = 0):
    '''
    Time delay addition
    <<Input>>
    MArray      ... Microphone Array(AS Format)
    Channel     ... Channel Num.
    tau         ... Time delay
    <<Output>>
    None
    '''
    ############################
    ### --- Format check --- ###
    ############################
    if MArray.Format is 'BSS':
        print("Warning(ASP.Delay) : Format of input MArray is converted into 'AS' format.")
        MArray.ConvForm('AS')
    else:
        pass
    
    ##################################
    ### --- Parameter Settings --- ###
    ##################################
    FFTL    = MArray.FFTL
    HFFTL   = np.ceil(FFTL / 2) + 1
    Fs      = MArray.Fs
    FVec    = np.array(range(FFTL), dtype = np.float64)
    FVec    = FVec / FFTL *Fs
    omega   = 2.0 * np.pi * FVec[0:HFFTL]
    Nmic    = MArray.Nmic

    if channel > Nmic:
        print("Error(ASP.Delay) : Selected channel is Not Exists")
        return None
    num_samples = MArray.NSample
    
    ####################
    ### --- Main --- ###
    ####################
    ### --- Memory allocation --- ###
    X         = MArray.Get()
    Y         = X.copy()
    (Nmic, Nfrm, HFFTL) = X.shape
    DM        = np.zeros((Nfrm, HFFTL), dtype = np.complex128)
    ### --- Delay matrix --- ###    
    for nfrm in range(Nfrm):
        DM[nfrm] = np.exp(-1j * omega[0:HFFTL] * tau / 2.0)
    
    ### --- Apply --- ###
    Y[channel,:,:] = Y[channel,:,:] * DM
    
    MArray.Set(Nfrm, num_samples, Nmic, Y)
    
    return None

def calc_delay_time(MArray, angle = 0):
    '''
    Calc delay time from angle

    :param MArray:  Microphone array
    :param angle:   Null angle [deg]
    :return:        Delay time [sec]
    '''

    mic_int = MArray.MicInt
    sound_v  = MArray.SV
    r_angle  = angle / 180.0 * np.pi
    delay    = mic_int * np.sin(r_angle) / sound_v
    
    return delay

def add_noise_segment(s, n, snr = 0.0):
    """
    Noise Addition (Segmental SNR)

    :param s:       Signal
    :param n:       Noise
    :param snr:     SNR [dB]
    :return:        Mixed waveform, Signal, and normalized noise
    """
    ####################
    ### --- Main --- ###
    ####################    
    ### --- RMS calc. --- ###
    s = np.float64(s)
    n = np.float64(n)
    rms_s = np.sqrt(np.mean(s ** 2))
    rms_n = np.sqrt(np.mean(n ** 2))
    
    ### --- Control Noise power --- ###
    n = n * (rms_s / rms_n)
    n = n * (10 ** (- snr / 20.0))
    n = np.array(n, dtype = sp.short)
    
    ### --- Adjust both length --- ###
    (s, n) = insert_zero(s, n, dtype = sp.short)
        
    ### --- Control Noise power --- ###
    y = np.array((s + n) / 2, dtype = sp.short)
    
    return y, s, n

def insert_zero(x1, x2, dtype = sp.short):
    """
    Insert zero elements to adjust array length

    :param x1:      Array 1
    :param x2:      Array 2
    :param dtype:   Data type
    :return:        Length-adjusted arrays
    """

    len1 = len(x1)
    len2 = len(x2)
    if len1 > len2:
        x2 = np.hstack((x2, np.zeros((len1 - len2), dtype)))
    elif len2 > len1:
        x1 = np.hstack((x1, np.zeros((len2 - len1), dtype)))
    else:
        pass
    
    return (x1, x2)


class MArray:
    """
    Microphone Array
    --------
    Modified on 2018/04/04
    @author: Naohiro TAWARA (KOBAYASHI LAb., Waseda Univ.)

    Originaly created on 2013/09/04
    @author: Motoi OMACHI(KOBAYASHI Lab., Waseda Univ.)
    --------
    """
    
    def __init__(self, Spec, MicInt = 0.03, SV = 340.0):
        '''
        Initialization of Spec class

        :param Spec: Spectrogram structure
        :param MicInt: Microphone Interval [m]
        :param SV: Sound velocity [m/s]
        '''

        #################################
        ### --- Inherit from Spec --- ###
        #################################
        self.FrameSize = Spec.FrameSize
        self.FrameShift = Spec.FrameShift
        self.FFTL = Spec.FFTL
        self.Fs = Spec.Fs
        self.WindowType = Spec.WindowType
        self.Format = Spec.Format
        self.Nfrm = Spec.Nfrm
        self.NSample = Spec.NSample
        self.Nmic =Spec.Nmic
        self.Data = Spec.Data
        
        ######################
        ### --- MArray --- ###
        ######################
        self.MicInt = MicInt
        self.SV = SV
        
        return None

    def show_info(self, Detail = False):
        """
        Display Information about Local Feature Extraction
        <<Input>>
        Detail    : Display parameter in detail
        <<Output>>
        None
        """
        print("================ MArray =================")
        print("FrameSize       : {}".format(str(self.FrameSize)))
        print("FrameShift      : {}".format(str(self.FrameShift)))
        print("FFTL            : {}".format(str(self.FFTL)))
        print("Nfrm            : {]".format(str(self.Nfrm)))
        print("Fs              : {}".format(str(self.Fs)))
        print("WindowType      : {}".format(self.WindowType))
        if Detail:
            print("NSample         : {}".format((self.NSample)))
        print("Nmic            : {}".format(str(self.Nmic)))
        print("Mic. Interval   : {}".format(str(self.MicInt)))
        print("Sound Vel.      : {}".format(str(self.SV)))
        if self.Data is None:
            print("Format      : None")
            print("Component   : None")
        else:
            print("Format      : {}".format(self.Format))
            print("Component   : [{}, {}]".format(str(self.Data.shape), str(self.Data.dtype)))
            if Detail:
                print(self.Data)
        print("=======================================")
    
    def get(self, Form = 'Complex'):
        '''
        Get Value
        <<Input>>
        Form    : Spectrogram Format
                  * 'Complex'   : Complex Spectrum
                  * 'Amplitude' : Amplitude Spectrum
                  * 'Phase'     : Phase Spectrum
        <<Output>>
        Data    : Local Feature
        '''
        X = self.Data
        if Form is 'Complex':
            pass
        elif Form is 'Amplitude':
            X = abs(X)
        elif Form is 'Phase':
            X = X/np.abs(X)
        else:
            print('Error(ASP.MArray.Get): Format Error')
        
        return X
    
    def set(self, Nfrm, NSample, Nmic, Data):
        """
        Set Data
        <<Input>>
        Nfrm        : Num. of frames
        NSample     : Num. of Samples
        Nmic        : Num. of Michrophone
        Data        : Spectrogram
        <<Output>>
        None
        """
        self.Nfrm = Nfrm
        self.NSample = NSample
        self.Nmic = Nmic
        self.Data =Data
    
    def Copy(self):
        '''
        Clone this class
        <<Input>>
        None
        <<Output>>
        None
        '''
        return cp.copy(self)
    
    def destroy(self, Format='AS'):
        '''
        Reset Data and Format
        <<Input>>
        Format    : Format
        <<Output>>
        None
        '''
        self.Nfrm = 0
        self.NSample = 0
        self.Nmic = 0
        #self.Data
        self.Format = Format
        
        return None
    
    def convert_form(self, Form='BSS'):
        '''
        Convert Format
        <<Input>>
        Form      : Target format
                    * 'BSS'     ... Analysis-Synthesis format -> BSS format
                    * 'AS'      ... BSS format -> Analysis-Synthesis format
        <<Output>>
        None
        '''
        ##################################
        ### --- Parameter Settings --- ###
        ##################################
        Nmic = self.Nmic
        Nfrm = self.Nfrm
        FFTL = self.FFTL
        HFFTL = int(np.ceil(FFTL/2)+1)
        
        ####################
        ### --- Main --- ###
        ####################
        X = self.Data
        if Form is 'BSS':
            # Format check
            if self.Format is 'AS':
                pass
            else:
                print("Error(ASP.MArray.ConvForm): Input Format is not correct")
                return 0
            # Memory allocation
            Y = sp.zeros((HFFTL,Nmic,Nfrm),dtype=np.complex128)
            # Update
            for mic in range(Nmic):
                X_= X[mic]
                for frm in range(Nfrm):
                    X_frm = X_[frm]
                    for freq in range(HFFTL):
                        Y[freq,mic,frm] = X_frm[freq]
            self.Format = 'BSS'
            self.Data = Y
        elif Form is 'AS':
            # Format check
            if self.Format is 'BSS':
                pass
            else:
                print("Error(ASP.MArray.ConvForm): Input Format is not correct")
                return 0
            # Memory allocation
            Y = sp.zeros((Nmic, Nfrm, HFFTL), dtype = np.complex128)
            # Update
            for freq in range(HFFTL):
                X_freq = X[freq]
                for nmic in range(Nmic):
                    X_m = X_freq[nmic]
                    for nfrm in range(Nfrm):
                        Y[nmic, nfrm, freq] = X_m[nfrm]
            self.Format='AS'
            self.Data=Y
=======
import numpy as np
import scipy as sp
import copy  as cp
import Util  as u
import STFT  as stft

'''
Utility functions for Acoustic Signal Processing
--------
Created on 2013/09/06
@author: Motoi OMACHI(KOBAYASHI Lab., Waseda Univ.)
--------
'''
MinV = 1e-10

def beam_former(MArray, channels = (0,1), angle=90, shape='Card'):
    '''
    Cardioid-shaped Beamformer
    <<Input>>
    MArray      ... Microphone Array(BSS Format)
    Ch          ... Channel
    Angle       ... Null Angle [deg]
    Shape       ... Beam shape
                    * 'Card'   : Cardioid shaped
                    * 'Dipole' : Dipole shaped
    <<Output>>
    OutBF       ... Beamformer output
    '''
    ##################################
    ### --- Parameter Settings --- ###
    ##################################
    if MArray.Format is 'BSS':
        MArray.ConvForm('AS')
    else:
        pass
    
    tau = calc_delay_time(MArray, angle)
    
    ####################
    ### --- Main --- ###
    ####################
    if shape is 'Card':
        add_delay(MArray, tau, channels[0])
    elif shape is 'Dipole':
        pass
    else:
        print("Warning(ASP.BeamFormer) : Beam shape '{}' is not defined.".format(str(shape)))
        add_delay(MArray, tau, channels[0])
    
    X = MArray.get()
    
    Y = X[channels[1]] - X[channels[0]]
#     (Nmic,HFFTL,Nfrm)=X.shape
    
    return Y

def add_delay(MArray, tau = 0.0, channel = 0):
    '''
    Time delay addition
    <<Input>>
    MArray      ... Microphone Array(AS Format)
    Channel     ... Channel Num.
    tau         ... Time delay
    <<Output>>
    None
    '''
    ############################
    ### --- Format check --- ###
    ############################
    if MArray.Format is 'BSS':
        print("Warning(ASP.Delay) : Format of input MArray is converted into 'AS' format.")
        MArray.ConvForm('AS')
    else:
        pass
    
    ##################################
    ### --- Parameter Settings --- ###
    ##################################
    FFTL    = MArray.FFTL
    HFFTL   = np.ceil(FFTL / 2) + 1
    Fs      = MArray.Fs
    FVec    = np.array(range(FFTL), dtype = np.float64)
    FVec    = FVec / FFTL *Fs
    omega   = 2.0 * np.pi * FVec[0:HFFTL]
    Nmic    = MArray.Nmic

    if channel > Nmic:
        print("Error(ASP.Delay) : Selected channel is Not Exists")
        return None
    num_samples = MArray.NSample
    
    ####################
    ### --- Main --- ###
    ####################
    ### --- Memory allocation --- ###
    X         = MArray.Get()
    Y         = X.copy()
    (Nmic, Nfrm, HFFTL) = X.shape
    DM        = np.zeros((Nfrm, HFFTL), dtype = np.complex128)
    ### --- Delay matrix --- ###    
    for nfrm in range(Nfrm):
        DM[nfrm] = np.exp(-1j * omega[0:HFFTL] * tau / 2.0)
    
    ### --- Apply --- ###
    Y[channel,:,:] = Y[channel,:,:] * DM
    
    MArray.Set(Nfrm, num_samples, Nmic, Y)
    
    return None

def calc_delay_time(MArray, angle = 0):
    '''
    Calc delay time from angle

    :param MArray:  Microphone array
    :param angle:   Null angle [deg]
    :return:        Delay time [sec]
    '''

    mic_int = MArray.MicInt
    sound_v  = MArray.SV
    r_angle  = angle / 180.0 * np.pi
    delay    = mic_int * np.sin(r_angle) / sound_v
    
    return delay

def add_noise_segment(s, n, snr = 0.0):
    """
    Noise Addition (Segmental SNR)

    :param s:       Signal
    :param n:       Noise
    :param snr:     SNR [dB]
    :return:        Mixed waveform, Signal, and normalized noise
    """
    ####################
    ### --- Main --- ###
    ####################    
    ### --- RMS calc. --- ###
    s = np.float64(s)
    n = np.float64(n)
    rms_s = np.sqrt(np.mean(s ** 2))
    rms_n = np.sqrt(np.mean(n ** 2))
    
    ### --- Control Noise power --- ###
    n = n * (rms_s / rms_n)
    n = n * (10 ** (- snr / 20.0))
    n = np.array(n, dtype = sp.short)
    
    ### --- Adjust both length --- ###
    (s, n) = insert_zero(s, n, dtype = sp.short)
        
    ### --- Control Noise power --- ###
    y = np.array((s + n) / 2, dtype = sp.short)
    
    return y, s, n

def insert_zero(x1, x2, dtype = sp.short):
    """
    Insert zero elements to adjust array length

    :param x1:      Array 1
    :param x2:      Array 2
    :param dtype:   Data type
    :return:        Length-adjusted arrays
    """

    len1 = len(x1)
    len2 = len(x2)
    if len1 > len2:
        x2 = np.hstack((x2, np.zeros((len1 - len2), dtype)))
    elif len2 > len1:
        x1 = np.hstack((x1, np.zeros((len2 - len1), dtype)))
    else:
        pass
    
    return (x1, x2)


class MArray:
    """
    Microphone Array
    --------
    Modified on 2018/04/04
    @author: Naohiro TAWARA (KOBAYASHI LAb., Waseda Univ.)

    Originaly created on 2013/09/04
    @author: Motoi OMACHI(KOBAYASHI Lab., Waseda Univ.)
    --------
    """
    
    def __init__(self, Spec, MicInt = 0.03, SV = 340.0):
        '''
        Initialization of Spec class

        :param Spec: Spectrogram structure
        :param MicInt: Microphone Interval [m]
        :param SV: Sound velocity [m/s]
        '''

        #################################
        ### --- Inherit from Spec --- ###
        #################################
        self.FrameSize = Spec.FrameSize
        self.FrameShift = Spec.FrameShift
        self.FFTL = Spec.FFTL
        self.Fs = Spec.Fs
        self.WindowType = Spec.WindowType
        self.Format = Spec.Format
        self.Nfrm = Spec.Nfrm
        self.NSample = Spec.NSample
        self.Nmic =Spec.Nmic
        self.Data = Spec.Data
        
        ######################
        ### --- MArray --- ###
        ######################
        self.MicInt = MicInt
        self.SV = SV
        
        return None

    def show_info(self, Detail = False):
        """
        Display Information about Local Feature Extraction
        <<Input>>
        Detail    : Display parameter in detail
        <<Output>>
        None
        """
        print("================ MArray =================")
        print("FrameSize       : {}".format(str(self.FrameSize)))
        print("FrameShift      : {}".format(str(self.FrameShift)))
        print("FFTL            : {}".format(str(self.FFTL)))
        print("Nfrm            : {]".format(str(self.Nfrm)))
        print("Fs              : {}".format(str(self.Fs)))
        print("WindowType      : {}".format(self.WindowType))
        if Detail:
            print("NSample         : {}".format((self.NSample)))
        print("Nmic            : {}".format(str(self.Nmic)))
        print("Mic. Interval   : {}".format(str(self.MicInt)))
        print("Sound Vel.      : {}".format(str(self.SV)))
        if self.Data is None:
            print("Format      : None")
            print("Component   : None")
        else:
            print("Format      : {}".format(self.Format))
            print("Component   : [{}, {}]".format(str(self.Data.shape), str(self.Data.dtype)))
            if Detail:
                print(self.Data)
        print("=======================================")
    
    def get(self, Form = 'Complex'):
        '''
        Get Value
        <<Input>>
        Form    : Spectrogram Format
                  * 'Complex'   : Complex Spectrum
                  * 'Amplitude' : Amplitude Spectrum
                  * 'Phase'     : Phase Spectrum
        <<Output>>
        Data    : Local Feature
        '''
        X = self.Data
        if Form is 'Complex':
            pass
        elif Form is 'Amplitude':
            X = abs(X)
        elif Form is 'Phase':
            X = X/np.abs(X)
        else:
            print('Error(ASP.MArray.Get): Format Error')
        
        return X
    
    def set(self, Nfrm, NSample, Nmic, Data):
        """
        Set Data
        <<Input>>
        Nfrm        : Num. of frames
        NSample     : Num. of Samples
        Nmic        : Num. of Michrophone
        Data        : Spectrogram
        <<Output>>
        None
        """
        self.Nfrm = Nfrm
        self.NSample = NSample
        self.Nmic = Nmic
        self.Data =Data
    
    def Copy(self):
        '''
        Clone this class
        <<Input>>
        None
        <<Output>>
        None
        '''
        return cp.copy(self)
    
    def destroy(self, Format='AS'):
        '''
        Reset Data and Format
        <<Input>>
        Format    : Format
        <<Output>>
        None
        '''
        self.Nfrm = 0
        self.NSample = 0
        self.Nmic = 0
        #self.Data
        self.Format = Format
        
        return None
    
    def convert_form(self, Form='BSS'):
        '''
        Convert Format
        <<Input>>
        Form      : Target format
                    * 'BSS'     ... Analysis-Synthesis format -> BSS format
                    * 'AS'      ... BSS format -> Analysis-Synthesis format
        <<Output>>
        None
        '''
        ##################################
        ### --- Parameter Settings --- ###
        ##################################
        Nmic = self.Nmic
        Nfrm = self.Nfrm
        FFTL = self.FFTL
        HFFTL = int(np.ceil(FFTL/2)+1)
        
        ####################
        ### --- Main --- ###
        ####################
        X = self.Data
        if Form is 'BSS':
            # Format check
            if self.Format is 'AS':
                pass
            else:
                print("Error(ASP.MArray.ConvForm): Input Format is not correct")
                return 0
            # Memory allocation
            Y = sp.zeros((HFFTL,Nmic,Nfrm),dtype=np.complex128)
            # Update
            for mic in range(Nmic):
                X_= X[mic]
                for frm in range(Nfrm):
                    X_frm = X_[frm]
                    for freq in range(HFFTL):
                        Y[freq,mic,frm] = X_frm[freq]
            self.Format = 'BSS'
            self.Data = Y
        elif Form is 'AS':
            # Format check
            if self.Format is 'BSS':
                pass
            else:
                print("Error(ASP.MArray.ConvForm): Input Format is not correct")
                return 0
            # Memory allocation
            Y = sp.zeros((Nmic, Nfrm, HFFTL), dtype = np.complex128)
            # Update
            for freq in range(HFFTL):
                X_freq = X[freq]
                for nmic in range(Nmic):
                    X_m = X_freq[nmic]
                    for nfrm in range(Nfrm):
                        Y[nmic, nfrm, freq] = X_m[nfrm]
            self.Format='AS'
            self.Data=Y
>>>>>>> d8c5583da6d9f9781cca014d1f3fc63614f80e5f
