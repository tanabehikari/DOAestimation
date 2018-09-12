import sys
import numpy as np
import scipy as sp
import wave
import datetime
import os

#import matplotlib.pyplot as plt

'''
Utility functions
--------
Created on 2013/09/04
@author: Motoi OMACHI(KOBAYASHI Lab., Waseda Univ.)
--------
'''
    
MinV=1e-10

def RawRead(FNAME,Form='short'):
    '''
    Load RAW-type Format file
    <<Input>>
    FNAME  ... File Name
    Form   ... Format
               * 'short'  ; short
               * 'double' : double
    <<Output>>
    None
    '''
    if Form is 'short':
        type=np.short
    elif Form is 'double':
        type=np.double

    try:
        x=np.fromfile(FNAME,type)
    except:
        print "(RawRead)FILE I/O error!: "+FNAME
        sys.exit()
    return x

def RawWrite(FNAME,x,Norm=False):
    '''
    Save wavefrom as Raw-type Format file
    <<Input>>
    FNAME  ... File Name
    x      ... Waveform
    Norm   ... Normalization
    <<Output>>
    None
    '''
    if Norm:
        x=np.array(x/max(x)*(2**15),dtype=np.short)
    else:
        x=np.array(x,dtype=np.short)
    
    try:
        x.tofile(FNAME)
    except:
        print "(RawWrite)FILE I/O error!: "+FNAME
        sys.exit()
    return 0

def WavRead(FNAME,dtype=np.float64):
    '''
    Load WAVE Format file
    <<Input>>
    FNAME  ... File Name
    dtype  ... Data type
    <<Output>>
    x      ... Waveform
    '''
    ### File Open ###
    try:
        wf=wave.open(FNAME,'rb')
    except:
        print "FILE I/O error!"
        sys.exit()
    
    ### Load all data ###
    data = wf.readframes(wf.getnframes())
    x    = sp.fromstring(data,sp.int16)
    
    wf.close()
    
    return x

def Whitening(x):
    '''
    Whitening
    <<Input>>
    x    ... Data
    <<Output>>
    y    ... Whitened data
    P    ... Projection matrix
    '''
    Dim,NSample=x.shape
    R=np.dot(x,x.T)/NSample
    e,V = np.linalg.eig(R)
    #idx = np.argsort(abs(e))[::-1]
    #e   = e[idx]
    #V   = V[:,idx]
    E=np.diag(e)
    P=np.dot(np.sqrt(np.linalg.inv(E)),V.T)
    y=np.dot(P,x)
    
    return y,P

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

def Standardize(x):
    '''
    Norm Normalization
    <<Input>>
    x    ... Data
    <<Output>>
    y    ... Normalized data
    Norm ... Norm
    '''
    Dim,NSample=x.shape
    x_mu=np.mean(x,axis=1).reshape(Dim,1)
    x_sd=np.std(x,axis=1).reshape(Dim,1)
    y=(x-x_mu)/x_sd
    
    return y,(x_mu,x_sd)

def Standardize_Power(x_):
    '''
    Norm Normalization
    <<Input>>
    x    ... Data(Complex Value)
    <<Output>>
    y    ... Normalized data
    Norm ... Norm
    '''
    Dim,NSample=x_.shape
    x=abs(x_)
    x_p=x/abs(x)
    x_mu=np.mean(x,axis=1).reshape(Dim,1)
    x_sd=np.std(x,axis=1).reshape(Dim,1)
    y=(x-x_mu)/x_sd
    y=abs(y)*x_p
    
    return y,(x_mu,x_sd)


def Window1(WindowSize=10,WindowType='hanning'):
    '''
    Window function
    <<Input>>
    WindowSize   ... WindowSize
    WindowType   ... Window Type
                     * 'Hanning'   : Hanning Window
                     * 'Hanning2'  : Hanning Window (pi delayed)
                     * 'Rectangle' : Tectangular window
    <<Output>>
    window       ... Window
    '''
    if WindowType is 'hanning':
        t=np.array(range(0,WindowSize),dtype=np.float64)
        window=0.5-0.5*np.cos(2*np.pi*t/WindowSize)
    elif WindowType is 'hanning2':
        t=np.array(range(0,WindowSize),dtype=np.float64)
        window=0.5-0.5*np.cos(2*np.pi*t/WindowSize+np.pi)
    elif WindowType is 'hamming':
        t=np.array(range(0,WindowSize),dtype=np.float64)
        window=0.54-0.46*np.cos(2*np.pi*t/WindowSize)
    elif WindowType is 'rectangle':
        window=sp.ones(WindowSize)
    else:
        print(WindowType + " is not supported window type.")
        print("hanning window is used.")
        window=sp.hanning(WindowSize)
    return window

def Window2(YSize=10,XSize=10,WindowPos=''):
    '''
    2D-Window function
    <<Input>>
    YSize       ... Window Size for Y-axis
    XSize       ... Window Size for X-axis
    WindowPos   ... Position of Window
    <<Output>>
    window      ... Window
    '''
    windowX=Window1(XSize,'Hanning')
    windowY=Window1(YSize,'Hanning')
    if WindowPos=='LT':
        windowX[0:np.ceil(XSize/2)]=1
        windowY[0:np.ceil(YSize/2)]=1
    elif WindowPos=='T':
        windowY[0:np.ceil(YSize/2)]=1
    elif WindowPos=='RT':
        windowY[0:np.ceil(YSize/2)]=1
        windowX[np.ceil(XSize/2):]=1
    elif WindowPos=='R':
        windowX[np.ceil(XSize/2):]=1
    elif WindowPos=='RD':
        windowX[np.ceil(XSize/2):]=1
        windowY[np.ceil(YSize/2):]=1
    elif WindowPos=='D':
        windowY[np.ceil(YSize/2):]=1
    elif WindowPos=='LD':
        windowY[np.ceil(YSize/2):]=1
        windowX[0:np.ceil(XSize/2)]=1
    elif WindowPos=='L':
        windowX[0:np.ceil(XSize/2)]=1
    elif WindowPos=='LT2':
        windowX[0:XSize]=1
        windowY[0:np.ceil(YSize/2)]=1
    elif WindowPos=='L2':
        windowX[0:XSize]=1
    elif WindowPos=='LD2':
        windowY[np.ceil(YSize/2):]=1
        windowX[0:XSize]=1
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
    Wx=sp.zeros((YSize,XSize),dtype=np.float64)
    for ix in range(YSize):
        Wx[ix]=windowX
    Wy=sp.zeros((XSize,YSize),dtype=np.float64)
    for iy in range(XSize):
        Wy[iy]=windowY
    window=Wx * Wy.T
    return window

def ReplaceMinV(M,MinV):
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
        YSize,XSize=M.shape
        if isinstance(XSize,tuple):
            print "Error(LocalFeqtureExt) : Unsupported array is input"
            sys.exit()
    except:
        # Vector
        XSize=M.shape
        XSize=XSize[0]
        YSize=1
    
    for iy in range(YSize):
        if YSize>1:
            M_=M[iy]
        else:
            M_=M
        # Comparison
        for ix in range(XSize):
            if M_[ix]<=MinV:
                M_[ix]=MinV
            else:
                pass
        if YSize>1:
            M[iy]=M_
        else:
            M=M_
    return M

def ReplaceMaxV(M,MaxV):
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
        YSize,XSize=M.shape
        if isinstance(XSize,tuple):
            print "Error(LocalFeqtureExt) : Unsupported array is input"
            sys.exit()
    except:
        # Vector
        XSize=M.shape
        XSize=XSize[0]
        YSize=1
    
    for iy in range(YSize):
        if YSize>1:
            M_=M[iy]
        else:
            M_=M
        # Comparison
        for ix in range(XSize):
            if M_[ix]>=MaxV:
                M_[ix]=MaxV
            else:
                pass
        if YSize>1:
            M[iy]=M_
        else:
            M=M_
    return M

def Sigmoid(x):
    '''
    Sigmoid function
    <<Input>>
    x        ...  Data
    <<Output>>
    y        ... value
    '''
    x_=np.exp(-x)
    return 1.0/(1.0+x_)

def ConvMatForm(X,Form='BSS'):
    '''
    Convert Matrix Format
    <<Input>>
    X         : Matrix
    Form      : Target format
                * 'BSS'     ... Analysis-Synthesis format -> BSS format
                * 'AS'      ... BSS format -> Analysis-Synthesis format
    <<Output>>
    Y         : Converted Matrix
    '''
    ##################################
    ### --- Parameter Settings --- ###
    ##################################
    if Form is 'BSS':
        (Nmic,Nfrm,HFFTL)=X.shape
    else:
        (HFFTL,Nmic,Nfrm)=X.shape
    
    ####################
    ### --- Main --- ###
    ####################
    if Form is 'BSS':
        # Memory allocation
        Y=sp.zeros((HFFTL,Nmic,Nfrm),dtype=np.complex128)
        # Update
        for mic in range(Nmic):
            X_=X[mic]
            for frm in range(Nfrm):
                X_frm=X_[frm]
                for freq in range(HFFTL):
                    Y[freq,mic,frm]=X_frm[freq]
    
    elif Form is 'AS':
        # Memory allocation
        Y=sp.zeros((Nmic,Nfrm,HFFTL),dtype=np.complex128)
        # Update
        for freq in range(HFFTL):
            X_freq=X[freq]
            for nmic in range(Nmic):
                X_m=X_freq[nmic]
                for nfrm in range(Nfrm):
                    Y[nmic,nfrm,freq]=X_m[nfrm]
                    
    return Y

def CalcQScore(x,y):
    '''
    Calcuration of Correation coefficient
    <<Input>>
    x      ... Vector
    y      ... Vector
    <<Output>>
    QScore ... Q-Score
    Q      ... Corelation coefficient
    '''
    ### --- Calc. Correlation coefficient --- ###
    ndim,nsample=x.shape
    x_mu  = np.mean(x,axis=1).reshape((ndim,1))
    x_    = x - np.dot(x_mu,np.ones((1,nsample)))
    y_mu  = np.mean(y,axis=1).reshape((ndim,1))
    y_    = y - np.dot(y_mu,np.ones((1,nsample)))
    nume  = np.dot(x_,y_.T)
    dR_x  = np.diag(np.dot(x_,x_.T)).reshape((ndim,1))
    dR_y  = np.diag(np.dot(y_,y_.T)).reshape((ndim,1))
    deno  = np.sqrt(np.dot(dR_x,dR_y.T))
    Q     = nume/deno
    
    ### --- Calc. Q-Score --- ###
    diagQ  = np.sum(np.diag(Q))
    odiagQ = np.sum((1-np.eye(ndim))*Q)
    QScore = diagQ - odiagQ
    
    return Q,QScore

def GetDateTime():
    '''
    Get Date and Time
    <<Input>>
    NONE
    <<Output>>
    DT    ... Date and Time
    '''
    ### --- Get Date and Time --- ###
    d=datetime.datetime.today()
    ### --- Year --- ###
    d_ye=str(d.year)
    ### --- Month --- ###
    d_mo=str(d.month)
    if d.month<10:
        d_mo='0'+d_mo
    ### --- Day --- ###
    d_da=str(d.day)
    if d.day<10:
        d_da='0'+d_da
    ### --- Hour --- ###
    d_ho=str(d.hour)
    if d.hour<10:
        d_ho='0'+d_ho
    ### --- Min. --- ###    
    d_mi=str(d.minute)
    if d.minute<10:
        d_mi='0'+d_mi
    
    return d_ye+d_mo+d_da+d_ho+d_mi

def mkdir(DIRNAME):
    '''
    Make Directory if it does not exist
    <<Input>>>
    DIRNAME    ... Directory name
    <<Output>>>
    None
    '''
    if os.path.exists(DIRNAME):
        pass
    else:
        os.mkdir(DIRNAME)

    return None

def CalcPower(x,FSize=1024,FShift=256):
    '''
    Power calcuration
    <<Input>>
    x      ... Waveform
    FSize  ... Frame size
    FShift ... Frame shift
    <<Output>>
    P      ... Power
    '''
    ##################################
    ### --- Parameter Settings --- ###
    ##################################
    x=np.float64(x)
    NSample=len(x)
    NFrames=int(np.ceil((NSample-FSize)/FShift)+1)
    
    ####################
    ### --- Main --- ###
    ####################
    ### --- Memory allocation --- ###
    P=np.zeros((NFrames,1),dtype=np.float64)
    T=np.zeros((NFrames,1),dtype=np.float64)
    ### --- Frame-wise processing --- ###
    bfrm=0
    for nfrm in range(NFrames):
        T[nfrm]=bfrm
	if bfrm+FSize>x.size:
            x_frm_=x[bfrm:x.size]
	    P[nfrm]=np.mean(x_frm**2)
        else:
            x_frm=x[bfrm:bfrm+FSize]
	    P[nfrm]=np.mean(x_frm*x_frm)
            bfrm += FShift
    return (P,T)

def CalcZC(x,FSize=1024,FShift=256):
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
    x=np.float64(x)
    NSample=len(x)
    NFrames=int(np.floor((NSample-FSize)/FShift)+1)
    
    ####################
    ### --- Main --- ###
    ####################
    ### --- Memory allocation --- ###
    ZC=np.zeros((NFrames,1),dtype=np.float64)
    ### --- Frame-wise processing --- ###
    bfrm=0
    for nfrm in range(NFrames):
	if bfrm+FSize>x.size:
            x_frm_=x[bfrm:x.size]
	    ZC[nfrm]=len(np.argwhere(x_frm[:-1]*x_frm[1:]<0))
        else:
            x_frm=x[bfrm:bfrm+FSize]
	    ZC[nfrm]=len(np.argwhere(x_frm[:-1]*x_frm[1:]<0))
            bfrm += FShift
    
    return ZC

def RAverage(x,a):
    '''
    Recursive average
    <<Input>>
    x      ... input samples
    a      ... lost factor
    <<Output>>
    y      ... averaged samples
    '''
    ##################################
    ### --- Parameter Settings --- ###
    ##################################
    x=np.complex128(x)
    NSample=len(x)
    
    ####################
    ### --- Main --- ###
    ####################
    ### --- Memory allocation --- ###
    y = np.zeros((NSample,),dtype=np.complex128)
    for i in range(NSample):
	if i == 0:
	    y[i] = x[i]
	else:
	    y[i] = a*y[i-1]+(1-a)*x[i]

    return y

if __name__ == "__main__" :
    pass
