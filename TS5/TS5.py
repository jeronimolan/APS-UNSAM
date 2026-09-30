#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 20:01:56 2026

@author: jeronimo
"""

#importar señal
###############################################################################
import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt


# Cargar el archivo CSV como un array de NumPy
# fs_audio, wav_data = sio.wavfile.read('prueba psd.wav')
# fs_audio, wav_data = sio.wavfile.read('silbido.wav')

#import sounddevice as sd
#sd.play(wav_data, fs_audio)

#plt.plot(wav_data)
###############################################################################

def plot_db_welch(subplot, data, fs):
    """
    Para que el código quede más legible, armo esta funcion que plotea la
    funcion que le paso por welch usando siempre la misma configuración
    """
    K = np.size(data)/1000
    L = np.size(data)/K
    ffs,DSP = scps.welch(x = data,fs = fs,window = 'hann_periodic',nperseg=L,nfft = 5000)
    DSP_dB = 10*np.log(DSP/ np.max(DSP))
    subplot.plot(ffs,DSP_dB, label = f"L = {L:.2f}")
    
    L2 = np.size(data)*(1/(10*K))
    ffs,DSP = scps.welch(x = data,fs = fs,window = 'hann_periodic',nperseg=L2,nfft = 5000)
    DSP_dB = 10*np.log(DSP / np.max(DSP))
    subplot.plot(ffs,DSP_dB, label = f"L = {L2:.2f}")

    L3 = np.size(data)*(1/(100*K))
    ffs,DSP = scps.welch(x = data,fs = fs,window = 'hann_periodic',nperseg=L3,nfft = 5000)
    DSP_dB = 10*np.log(DSP / np.max(DSP))
    subplot.plot(ffs,DSP_dB, label = f"L = {L3:.2f}")
    cucaracha.legend()




###############################################################################
##Señales de audio
import scipy.signal as scps
import scipy.io as sio
from scipy.io.wavfile import write


fig_audios, ((cucaracha, silbido, psd)) = plt.subplots(1,3, sharex=True)
fig_audios.suptitle("Estimación espectral de señales con Welch",y =1.02)
fig_audios.tight_layout(w_pad = 1)

cucaracha.set_title("Fragmento de \n 'la cucaracha' silbado")
cucaracha.set_xlabel("Frecuencia [Hz]")
cucaracha.set_ylabel("Densidad de Potencia [dB]")

fs_audio, wav_data = sio.wavfile.read('la cucaracha.wav')
plot_db_welch(cucaracha, wav_data, fs_audio)


psd.set_title("Prueba de psd")
psd.set_xlabel("Frecuencia [Hz]")
psd.set_ylabel("Densidad de Potencia [dB]")

fs_audio, wav_data = sio.wavfile.read('prueba psd.wav')
plot_db_welch(psd, wav_data, fs_audio)


silbido.set_title("Silbido")
silbido.set_xlabel("Frecuencia [Hz]")
silbido.set_ylabel("Densidad de Potencia [dB]")

fs_audio, wav_data = sio.wavfile.read('silbido.wav')
plot_db_welch(silbido, wav_data, fs_audio)

###############################################################################







