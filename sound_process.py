from python_speech_features import logfbank, delta
import numpy as np
import matplotlib.pyplot as plt

def smooth(x,window_len=15,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    np.hanning, np.hamming, np.bartlett, np.blackman, np.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def preprocess(sig,samplerate:int = 16000):
    # x_seq=np.arange(0,len(sig)/samplerate,1/samplerate)
    # plt.plot(x_seq,sig)
    # plt.show()
    # exit()
    # print(sig.shape)
    f_logfbank = logfbank(smooth(sig),16000, nfilt=40)
    f_logfbank = f_logfbank - (np.mean(f_logfbank, axis=0) +1e-8)
    delta_f = delta(f_logfbank,3)
    delta_delta_f = delta(delta_f,3)

    return np.concatenate((f_logfbank,delta_f,delta_delta_f),axis=1).astype(np.float32)

if __name__ == "__main__":
    print(preprocess(np.random.randint(-20,20,16000)))