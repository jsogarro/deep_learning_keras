"""
Sound Wave - sample file that shows how to load and plot wav files
"""
from scipy.io import wavfile
import matplotlib.pyplot as plt 

def main():
    # load our wav file
    rate, snd = wavfile.read(filename='../data/sms.wav')

    # plot the wav 
    plt.plot(snd)
    plt.title('SMS Sound Wav')
    plt.show()

    # show spectragram
    _ = plt.specgram(snd, NFFT=1024, Fs=44100
    plt.title('Wav Spectragam')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.show()

if __name__ == '__main__':
    main()