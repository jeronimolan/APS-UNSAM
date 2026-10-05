#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 20:01:56 2026

@author: jeronimo
"""

#importar señal
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scps
import scipy.io as sio

###############################################################################

def plot_db_welch(subplot, data, fs, n, K, P, BW, VAR):
    """
    Para que el código quede más legible, armo esta funcion que plotea la
    funcion que le paso por welch usando siempre la misma configuración
    """
    L = n/K
    ffs,DSP = scps.welch(x = data,fs = fs,window = 'hann_periodic',nperseg=L)
    DSP_dB = 10*np.log(DSP/ np.max(DSP))
    subplot.plot(ffs,DSP_dB, label = f"L = {L:.2f}")
    
    potencia = 0
    potencia_total = np.sum(DSP)
    
    for i in range(n):
        if (potencia > (potencia_total*P)):
            break
        potencia = potencia + DSP[i]
        BW[0] = i*(fs*K/n)
    VAR[0] = np.var(DSP)
    
    L2 = n*(1/(10*K))
    ffs,DSP = scps.welch(x = data,fs = fs,window = 'hann_periodic',nperseg=L2)
    DSP_dB = 10*np.log(DSP / np.max(DSP))
    subplot.plot(ffs,DSP_dB, label = f"L = {L2:.2f}")

    ##estimador de ancho de banda
    potencia = 0
    potencia_total = np.sum(DSP)
           
    for i in range(n):
        if (potencia > (potencia_total*P)):
            break
        potencia = potencia + DSP[i]
        BW[1] = i*(fs*10*K/n)
    VAR[1] = np.var(DSP)

    L3 = n*(1/(100*K))
    ffs,DSP = scps.welch(x = data,fs = fs,window = 'hann_periodic',nperseg=L3)
    DSP_dB = 10*np.log(DSP / np.max(DSP))
    subplot.plot(ffs,DSP_dB, label = f"L = {L3:.2f}")
    subplot.legend()
    
    potencia = 0
    potencia_total = np.sum(DSP)
    
    for i in range(n):
        if (potencia > (potencia_total*P)):
            break
        potencia = potencia + DSP[i]
        BW[2] = i*(fs*100*K/n)
    VAR[2] = np.var(DSP)
    
"""
###############################################################################
##Señales de audio
"""
##graficos en tiempo
fig_audios_t, ((cucaracha_t, silbido_t, psd_t)) = plt.subplots(1,3)
fig_audios_t.suptitle("Señales de audio a analizar en tiempo",y =1.02)
fig_audios_t.tight_layout(w_pad = 3)

cucaracha_t.set_title("Fragmento de \n 'la cucaracha' silbado")
cucaracha_t.set_xlabel("Tiempo [s]")
cucaracha_t.set_ylabel("Tension [V]")

psd_t.set_title("Fragmento de \n 'la cucaracha' silbado")
psd_t.set_xlabel("Tiempo [s]")
psd_t.set_ylabel("Tension [V]")

silbido_t.set_title("Fragmento de \n 'la cucaracha' silbado")
silbido_t.set_xlabel("Tiempo [s]")
silbido_t.set_ylabel("Tension [V]")



##graficos en frecuencia
fig_audios, ((cucaracha, silbido, psd)) = plt.subplots(1,3)
fig_audios.suptitle("Estimación espectral de señales con Welch",y =1.02)
fig_audios.tight_layout(w_pad = 1)

###para el grafico de tiempo

cucaracha.set_title("Fragmento de \n 'la cucaracha' silbado")
cucaracha.set_xlabel("Frecuencia [Hz]")
cucaracha.set_ylabel("Densidad de Potencia [dB]")

BW_cucaracha = [0,0,0]
VAR_cucaracha = [0,0,0]
fs_audio, wav_data = sio.wavfile.read('la cucaracha.wav')
plot_db_welch(cucaracha, wav_data, fs_audio, np.size(wav_data),144,0.95,BW_cucaracha,VAR_cucaracha)

###para el grafico de tiempo
cucaracha_t.plot(wav_data)

psd.set_title("Prueba de psd")
psd.set_xlabel("Frecuencia [Hz]")
psd.set_ylabel("Densidad de Potencia [dB]")

VAR_prueba = [0,0,0]
BW_prueba = [0,0,0]
fs_audio, wav_data = sio.wavfile.read('prueba psd.wav')
plot_db_welch(psd, wav_data, fs_audio, np.size(wav_data),144,0.95,BW_prueba,VAR_prueba)

###para el grafico de tiempo
psd_t.plot(wav_data)


silbido.set_title("Silbido")
silbido.set_xlabel("Frecuencia [Hz]")
silbido.set_ylabel("Densidad de Potencia [dB]")

VAR_silbido = [0,0,0]
BW_silbido = [0,0,0]
fs_audio, wav_data = sio.wavfile.read('silbido.wav')
plot_db_welch(silbido, wav_data, fs_audio, np.size(wav_data),144,0.96,BW_silbido,VAR_silbido)

###para el grafico de tiempo
silbido_t.plot(wav_data)


###############################################################################
##señales ECG

fs_ecg = 1000 # Hz

##################
## ECG con ruido
##################

# para listar las variables que hay en el archivo
sio.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./ECG_TP4.mat')

ecg_one_lead = mat_struct['ecg_lead']

hb_1_data = mat_struct['heartbeat_pattern1']
hb_2_data = mat_struct['heartbeat_pattern2']

ecg_data_lead = ecg_one_lead[5000:12000,0]
hb_1 = hb_1_data[:,0]
hb_2 = hb_2_data[:,0]


##graficos en tiempo
fig_ecgs_t, ((lead_t, hb_1_t, hb_2_t)) = plt.subplots(1,3)
fig_ecgs_t.suptitle("Señales de ECG a analizar en tiempo",y =1.02)
fig_ecgs_t.tight_layout(w_pad = 3)

lead_t.set_title("ECG lead")
lead_t.set_xlabel("Tiempo [s]")
lead_t.set_ylabel("Tension [V]")
lead_t.plot(ecg_data_lead)

hb_1_t.set_title("Patrón 1")
hb_1_t.set_xlabel("Tiempo [s]")
hb_1_t.set_ylabel("Tension [V]")
hb_1_t.plot(hb_1)

hb_2_t.set_title("Patrón 2")
hb_2_t.set_xlabel("Tiempo [s]")
hb_2_t.set_ylabel("Tension [V]")
hb_2_t.plot(hb_2)


##graficos en frecuencia
fig_ecgs, ((lead_w, hb_1_w, hb_2_w)) = plt.subplots(1,3)
fig_ecgs.suptitle("Estimación espectral de ECG con Welch",y =1.02)
fig_ecgs.tight_layout(w_pad = 1)


lead_w.set_title("ECG lead")
lead_w.set_xlabel("Frecuencia [Hz]")
lead_w.set_ylabel("Densidad de Potencia [dB]")

VAR_lead = [0,0,0]
BW_lead = [0,0,0]
plot_db_welch(lead_w, ecg_data_lead, fs_ecg, np.size(ecg_data_lead), 7, 0.95, BW_lead,VAR_lead) 

hb_1_w.set_title("Patrón 1")
hb_1_w.set_xlabel("Frecuencia [Hz]")
hb_1_w.set_ylabel("Densidad de Potencia [dB]")

VAR_hb1 = [0,0,0]
BW_hb1 = [0,0,0]
plot_db_welch(hb_1_w, hb_1, fs_ecg, np.size(hb_1), 1, 0.95, BW_hb1,VAR_hb1)

hb_2_w.set_title("Patrón 2")
hb_2_w.set_xlabel("Frecuencia [Hz]")
hb_2_w.set_ylabel("Densidad de Potencia [dB]")

VAR_hb2 = [0,0,0]
BW_hb2 = [0,0,0]
plot_db_welch(hb_2_w, hb_2, fs_ecg, np.size(hb_2), 1, 0.95, BW_hb2,VAR_hb2)

###############################################################################

fs_ppg = 400 # Hz

ppg = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)  # Omitir la cabecera si existe

fig_ppgs, ((ppg_t, ppg_w)) = plt.subplots(1,2)
fig_ppgs.suptitle("Señal de PPG a analizar en tiempo y frecuencia",y =1.02)
fig_ppgs.tight_layout(w_pad = 3)

ppg_t.set_title("PPG con ruido")
ppg_t.set_xlabel("Tiempo [s]")
ppg_t.set_ylabel("Tension [V]")
ppg_t.plot(ppg)

ppg_w.set_title("Espectro PPG con ruido")
ppg_w.set_xlabel("Frecuencia [Hz]")
ppg_w.set_ylabel("Densidad de Potencia [dB/Hz]")

VAR_ppg = [0,0,0]
BW_ppg = [0,0,0]
plot_db_welch(ppg_w, ppg, fs_ppg, np.size(ppg), 45, 0.95, BW_ppg,VAR_ppg)


###############################################################################

##tabla comparativa##




datos_sonidos = [(BW_cucaracha),(VAR_cucaracha),(BW_prueba),(VAR_prueba),(BW_silbido),(VAR_silbido)]
datos_ecg_lead = [(BW_lead),(VAR_lead)]
datos_ecg = [(BW_hb1),(VAR_hb1),(BW_hb2),(VAR_hb2)]
datos_ppg = [(BW_ppg),(VAR_ppg)]

##tabla sonidos
plt.figure(50,figsize=(8,2))
plt.title("Estimaciones de Ancho de banda \n según welch para distintos \n sonidos y K ")
tabla_a_10dB = plt.table(
    cellText=(datos_sonidos),
    colLabels=['K = 144','K = 1440','K = 14400'],
    rowLabels=['BW_cucaracha','VAR_cucaracha','BW_prueba','VAR_prueba','BW_silbido','VAR_silbido'],
    loc = 'center',
    cellLoc = 'center'
    )
plt.axis('off')
plt.tight_layout()
plt.show()

##tabla ecg
plt.figure(60,figsize=(8,2))
plt.title("Estimaciones de Ancho de banda \n según welch para distintos \n ECG y K ")
tabla_a_10dB = plt.table(
    cellText=(datos_ecg_lead),
    colLabels=['K = 7','K = 70','K = 700'],
    rowLabels=['BW_lead','VAR_lead'],
    loc = 'center',
    cellLoc = 'center'
    )
plt.axis('off')
plt.tight_layout()
plt.show()

plt.figure(70,figsize=(8,2))
plt.title("Estimaciones de Ancho de banda \n según welch para distintos \n ECG y K ")
tabla_a_10dB = plt.table(
    cellText=(datos_ecg),
    colLabels=['K = 1','K = 10','K = 100'],
    rowLabels=['BW_hb1','VAR_hb1','BW_hb2','VAR_hb2'],
    loc = 'center',
    cellLoc = 'center'
    )
plt.axis('off')
plt.tight_layout()
plt.show()

##tabla ppg
plt.figure(80,figsize=(8,2))
plt.title("Estimaciones de Ancho de banda \n según welch para PPG y K ")
tabla_a_10dB = plt.table(
    cellText=(datos_ppg),
    colLabels=['K = 45','K = 450','K = 4500'],
    rowLabels=['BW_ppg','VAR_ppg'],
    loc = 'center',
    cellLoc = 'center'
    )
plt.axis('off')
plt.tight_layout()
plt.show()

