import libs.utils as utils
import numpy as np
import libs.stft as stft

def Hermete(array):
    A = np.array(array)
    B = np.conjugate(A.T)
    return B


def MVbeamformer(wav1,wav2, deg,filename_out=None,micDistant = 0.03,soundVerocity = 340, fs = 16000):
    theta = np.pi*deg/180
    WAV1,param1 = stft.stft(wav1)
    WAV2,param2 = stft.stft(wav2)
    timeLength = WAV1.shape[0]  
    omegaLength = WAV1.shape[1]

    bpf_filter = stft.bpf(omegaLength, fs=16000, low_freq=300, up_freq=5499)
    bpf_filter = np.array(bpf_filter)
    WAV1 = bpf_filter *WAV1
    WAV2 = bpf_filter *WAV2

    for i in range(omegaLength):
        source1 = WAV1[:,i]
        source1 = np.array(source1,ndmin=2)
        source2 = WAV2[:,i]
        source2 = np.array(source2,ndmin=2)
        z_i = source1
        z_i = np.vstack((z_i,source2))
        z_i_gyou,z_i_retsu = z_i.shape[0],z_i.shape[1]

        zH_i = Hermete(z_i)
        R_i = np.dot(z_i,zH_i)/z_i_retsu
        omega = fs/((omegaLength-1)*2)*i 
        omega = 2*np.pi*omega 

        if R_i.all():#where R is not Zeros
            R_inv_i = np.linalg.inv(R_i)
            delay = micDistant*np.sin(theta)/soundVerocity 
            a_i = [1,np.exp(-1j*omega*delay)] ##array manifold vector##
            a_i = np.array(a_i,ndmin=2).T
            aH_i = Hermete(a_i)

            w_i = np.dot(R_inv_i,a_i)/(np.dot(np.dot(aH_i,R_inv_i),a_i))
            wH_i = Hermete(w_i)
            wH_i = np.array(wH_i)
        else:
            w_i = np.zeros((2,1))
            wH_i = np.zeros((1,2))
            
        Y_i = np.dot(wH_i,z_i)
        Y_T = np.array(Y_i[0],ndmin=2).T

        if i==0:
            Y1 = Y_T
            WH = wH_i    
        else:
            Y1 = np.hstack((Y1,Y_T))
            WH = np.vstack((WH,wH_i))

    y1 = stft.synth(Y1,param1)
#    utils.write__float32_as_int16(filename_out,16000,y1)
    W = Hermete(WH)
    return y1,W

def MVbeamformer2(wav1,wav2,deg):
	y1,W = MVbeamformer(wav1,wav2,deg)
	y2,W2 = MVbeamformer(wav1,wav2,deg+90)
    
	return y1,y2,W

