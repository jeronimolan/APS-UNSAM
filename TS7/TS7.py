#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 17:20:23 2026

@author: jeronimo
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scps

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

#h(n) = (1,1,1)
N = 1000
K = 1/3
w,rf  = scps.freqz((1,1,1),1,N)

#plt.plot(w,(K*rf))

#chev
b,a = scps.iirdesign(wp = np.pi/2,ws = np.pi*2/3,gpass = 3,gstop=40,output='ba',ftype='cheby1',fs = np.pi*2)
w, rf = scps.freqz(b, a, N)

rf = 20*np.log10(rf)

plt.plot(w,rf)


b,a = scps.iirdesign(wp = np.pi/2,ws = np.pi*55/100,gpass = 3,gstop=80,output='ba',ftype='cheby1',fs = np.pi*2)
w, rf = scps.freqz(b, a, N)

rf = 20*np.log10(rf)

plt.plot(w,rf)

#butter

b,a = scps.iirdesign(wp = np.pi/2,ws = np.pi*2/3,gpass = 3,gstop=40,output='ba',ftype='butter',fs = np.pi*2)
w, rf = scps.freqz(b, a, N)

rf = 20*np.log10(rf)

plt.plot(w,rf)


b,a = scps.iirdesign(wp = np.pi/2,ws = np.pi*55/100,gpass = 3,gstop=80,output='ba',ftype='butter',fs = np.pi*2)
w, rf = scps.freqz(b, a, N)

rf = 20*np.log10(rf)

plt.plot(w,rf)

#filtrado de ecg
##señales ECG

fs_ecg = 1000 # Hz

##################
## ECG con ruido
##################
import scipy.io as sio

# para listar las variables que hay en el archivo
sio.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./ECG_TP4.mat')

ecg_one_lead = mat_struct['ecg_lead']

ecg_one_lead = np.asarray(ecg_one_lead).squeeze()

ecg_data_lead = ecg_one_lead[5000:12000]

sos = scps.iirdesign(wp = 0.188,ws = 0.251,gpass = 3,gstop=60,output='sos',ftype='butter',fs = np.pi*2)
w, rf = scps.freqz_sos(sos, N)
"""
rf = 20*np.log10(rf)
plt.figure (2)
plt.plot(w,rf)

ecg_filtrada = scps.sosfilt(sos, ecg_data_lead)

plt.figure(10)
plt.title("Señales de ECG a analizar en tiempo",y =1.02)
plt.plot(ecg_filtrada, label = "filtrada")
plt.plot(ecg_data_lead, label = "original")
plt.legend()
plt.xlabel("Tiempo [s]")
plt.ylabel("Tension [V]")

################delta
delta_kronecker = np.zeros(7000)
delta_kronecker[0] = 1

r_i = scps.sosfilt(sos, delta_kronecker)

plt.figure(20)
plt.title("delta a analizar en tiempo",y =1.02)
plt.plot(r_i, label = "filtrada")
plt.plot(delta_kronecker , label = "original")
plt.legend()
plt.xlabel("Tiempo [s]")
plt.ylabel("Tension [V]")
"""


##%%...................Diseño de filtro................."

"""
y_min, y_max = -100, 5

def calc_resp( sos, leg_str, fig_n_axes):

    # Respuesta en frecuencia en Hz, de 0 a Nyquist
    f, h = scps.freqz_sos(sos, worN=8192, fs=fs_ecg)

    # Módulo en dB y fase desenrollada en radianes
    modulo_db = 20 * np.log10(np.maximum(np.abs(h), 1e-12))
    fase_rad = np.unwrap(np.angle(h))

    # Retardo de grupo: -d(fase)/d(omega), expresado en segundos
    omega = 2 * np.pi * f  # rad/s
    retardo_s = -np.gradient(fase_rad, omega)

    # En frecuencias donde el filtro prácticamente rechaza la señal,
    # la fase y el retardo pueden ser poco significativos.
    umbral_db = -80
    valido = modulo_db > umbral_db
    retardo_s_plot = np.where(valido, retardo_s, np.nan)

    (fig, (ax_mod, ax_fase)) = fig_n_axes

    # --- Módulo ---
    ax_mod.plot(f, modulo_db, linewidth=1.5,
                label= leg_str )
    ax_mod.set_ylim(y_min, y_max)
    ax_mod.set_ylabel("Módulo (dB)")
    ax_mod.set_title("Respuesta en frecuencia y plantilla")
    ax_mod.grid(True)

    # --- Fase ---
    ax_fase.plot(f, fase_rad, label= leg_str, linewidth=1.2)
    ax_fase.axvline(wp, color="gray", linestyle=":")
    ax_fase.axvline(ws, color="gray", linestyle=":")
    ax_fase.set_xlabel("Frecuencia (Hz)")
    ax_fase.set_ylabel("Fase (rad)")
    ax_fase.grid(True)

    plt.show()

    # Figura 2: retardo de grupo en segundos

    plt.figure(2, figsize=(10, 4))
    plt.plot(f, retardo_s_plot, label= leg_str)
    plt.axvline(wp, color="orange", linestyle="--", label=f"wp = {wp} Hz")
    plt.axvline(ws, color="red", linestyle="--", label=f"ws = {ws} Hz")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Retardo de grupo (s)")
    plt.title("Retardo de grupo del filtro IIR")
    plt.grid(True)
    plt.legend()

    plt.show()


plt.close('all')

fig_n_axes = plt.subplots(
    2, 1, sharex=True, num = 1, figsize=(10, 7), constrained_layout=True
)

calc_resp( sos_butt, 'butter', fig_n_axes )

calc_resp( sos_cheby2, 'cheby2', fig_n_axes )

calc_resp( sos_cauer, 'cauer', fig_n_axes )

# Ploteo de la plantilla

(fig, (ax_mod, ax_fase)) = fig_n_axes

# Banda pasante: zona no permitida por debajo de -ripple
rect_pasante = Rectangle(
    (0, y_min), wp, -ripple - y_min,
    facecolor="red", alpha=0.12,
    edgecolor="red", hatch="xxx", linewidth=1.2 , label = 'Plantilla'      
)
ax_mod.add_patch(rect_pasante)

# Banda de rechazo: zona no permitida por encima de -att
rect_rechazo = Rectangle(
    (ws, -att), fs_ecg / 2 - ws, y_max + att,
    facecolor="red", alpha=0.12,
    edgecolor="red", hatch="xxx", linewidth=1.2
)
ax_mod.add_patch(rect_rechazo)

# Límites de la plantilla
# ax_mod.axhline(-ripple, color="darkorange", linestyle="--",
#                label=f"Límite pasante: −{ripple} dB")
# ax_mod.axhline(-att, color="green", linestyle="--",
#                label=f"Límite rechazo: −{att} dB")
ax_mod.axvline(wp, color="gray", linestyle=":")
ax_mod.axvline(ws, color="gray", linestyle=":")

# Etiquetas de zonas
ax_mod.text(wp / 2, y_max - 6, "Banda pasante\npermitida: 0 a −3 dB",
            ha="center", va="top", fontsize=9)
ax_mod.text((ws + fs_ecg / 2) / 2, y_max - 6,
            "Banda de rechazo\npermitida: ≤ −40 dB",
            ha="center", va="top", fontsize=9)
ax_mod.text((wp + ws) / 2, y_min + 5, "Transición",
            ha="center", va="bottom", fontsize=9)

ax_mod.legend(loc="lower left", fontsize=8)

# Calculo la respuesta al impulso

delta_k = scps.unit_impulse(1000)
resp_imp_butt = scps.sosfiltfilt(sos_butt, delta_k)
resp_imp_cheby2 = scps.sosfiltfilt(sos_cheby2, delta_k)
resp_imp_cauer = scps.sosfiltfilt(sos_cauer, delta_k)

plt.figure(3)

plt.plot(resp_imp_butt, linewidth=1.25, label= 'butter' )
plt.plot(resp_imp_cheby2, linewidth=1.25, label= 'cheby2' )
plt.plot(resp_imp_cauer, linewidth=1.25, label= 'cauer' )
plt.legend()

plt.ylabel("Volts (V)")
plt.ylabel("Tiempo (ms)")
plt.title("Respuesta al impulso")

plt.close('all')
"""

from matplotlib.patches import Rectangle

wp = 35 #[Hz]
ws = 50 #[Hz]
Gpass = 3
Gstop = 40

ripple = Gpass
att = Gstop

sos_butt = scps.iirdesign(wp, ws, Gpass/2, Gstop/2, ftype='butter', fs = fs_ecg, output = 'sos')
sos_cheby2 = scps.iirdesign(wp, ws, Gpass/2, Gstop/2, ftype='cheby2', fs = fs_ecg, output = 'sos')
sos_cauer = scps.iirdesign(wp, ws, Gpass/2, Gstop/2, ftype='cauer', fs = fs_ecg, output = 'sos')

###################################
# Regiones de interés con ruido #
###################################
 
cant_muestras = ecg_one_lead.shape[0]

ECG_f_butt = scps.sosfilt(sos_butt, ecg_one_lead) 
ECG_f_cheb2 = scps.sosfilt(sos_cheby2, ecg_one_lead) 
ECG_f_cauer = scps.sosfilt(sos_cauer, ecg_one_lead) 

y_min, y_max = -100, 5
wp = 35 #[Hz]
ws = 50 #[Hz]

taps_win = 131
b_win_hann = scps.firwin2(taps_win, [0.0, wp, wp, fs_ecg/2], [1.0, 1.0, 0.0, 0.0], fs = fs_ecg, window= 'hann')   #me filtra la señal con fir
w,h = scps.freqz(b_win_hann,fs_ecg)            #veo la respuesta
rf = 20*np.log10(h) + 60
plt.figure (20)
plt.plot(w,rf)
taps_win = 131
b_win_hamming = scps.firwin2(taps_win, [0.0, wp, wp, fs_ecg/2], [1.0, 1.0, 0.0, 0.0], fs = fs_ecg, window= 'hamming')   #me filtra la señal con fir
w,h = scps.freqz(b_win_hamming,fs_ecg)            #veo la respuesta
rf = 20*np.log10(h) + 60
plt.figure (20)
plt.plot(w,rf)
taps_win = 191
b_win_boxcar = scps.firwin2(taps_win, [0.0, wp, wp, fs_ecg/2], [1.0, 1.0, 0.0, 0.0], fs = fs_ecg, window= 'boxcar')   #me filtra la señal con fir
w,h = scps.freqz(b_win_boxcar,fs_ecg)            #veo la respuesta
rf = 20*np.log10(h) + 60
plt.figure (20)
plt.plot(w,rf)
plt.ylim(y_min, y_max)

#scps.filtfilt(b, a, x)  #me saca la distorsion de fase pero no sirve a tiempo real
ECG_f_hann = scps.lfilter(b_win_hann,1,ecg_one_lead)
#ECG_f_hamming = lfilter(b,1,ecg...)
#ECG_f_boxcar = lfilter(b,1,ecg...)
#demora = (b_win_....shape[0] -1)//2


regs_interes = (
        [4000, 5500], # muestras
        [10e3, 11e3], # muestras
        )

for ii in regs_interes:
   
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
   
    plt.figure()
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    plt.plot(zoom_region, ECG_f_butt[zoom_region], label='Butterworth')
    plt.plot(zoom_region, ECG_f_cheb2[zoom_region], label='cheb2')
    plt.plot(zoom_region, ECG_f_cauer[zoom_region], label='cauer')
    plt.plot(zoom_region, ECG_f_hann[zoom_region], label='hann')
    #plt.plot(zoom_region, ECG_f_hamming[zoom_region], label='hamming')
    #plt.plot(zoom_region, ECG_f_boxcar[zoom_region], label='boxcar')
    
    plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
   
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
           
    plt.show()



""" 
#################################
# Regiones de interés sin ruido #
#################################

regs_interes = (
        np.array([5, 5.2]) *60*fs_ecg, # minutos a muestras
        np.array([12, 12.4]) *60*fs_ecg, # minutos a muestras
        np.array([15, 15.2]) *60*fs_ecg, # minutos a muestras
        )
 
for ii in regs_interes:
   
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
   
    plt.figure()
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    plt.plot(zoom_region, ECG_f_butt[zoom_region], label='Butterworth', linewidth=2)
    plt.plot(zoom_region, ECG_f_cheb2[zoom_region], label='cheb2', linewidth=2)
    plt.plot(zoom_region, ECG_f_cauer[zoom_region], label='cauer', linewidth=2)
   
    plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
   
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
           
    plt.show()
"""

#fir paga no distorsionar fase con retardo

#firls

plt.close('all')
###pasa banda

ws1 = .1
wp1 = 1.0
wp2 = 35
ws2 = 50

wpv = np.array([wp1,wp2]) * (2*np.pi) / fs_ecg
wsv = np.array([ws1,ws2]) * (2*np.pi) / fs_ecg

sos_butt = scps.iirdesign(wpv, wsv, Gpass/2, Gstop/2, ftype='butter', fs = fs_ecg, output = 'sos')
sos_cheby2 = scps.iirdesign(wpv, wsv, Gpass/2, Gstop/2, ftype='cheby2', fs = fs_ecg, output = 'sos')
sos_cauer = scps.iirdesign(wpv, wsv, Gpass/2, Gstop/2, ftype='cauer', fs = fs_ecg, output = 'sos')

taps_win = 2000

b_win_hann = scps.firwin2(taps_win, [0.0, ws1, wp1 , wp2, ws2, fs_ecg/2], [0.0, 0.0, 1.0, 1.0, 0.0, 0.0], fs = fs_ecg, window= 'hann') 

plt.figure(200)
f, h = scps.freqz_sos(sos_butt, fs=fs_ecg)
modulo_db = 20 * np.log10(np.maximum(np.abs(h), 1e-12))
fase_rad = np.unwrap(np.angle(h))
plt.plot(f,modulo_db)
f, h = scps.freqz_sos(sos_cauer, fs=fs_ecg)
modulo_db = 20 * np.log10(np.maximum(np.abs(h), 1e-12))
fase_rad = np.unwrap(np.angle(h))
plt.plot(f,modulo_db)
f, h = scps.freqz_sos(sos_cheby2, fs=fs_ecg)
modulo_db = 20 * np.log10(np.maximum(np.abs(h), 1e-12))
fase_rad = np.unwrap(np.angle(h))
plt.plot(f,modulo_db)

w,h = scps.freqz(b_win_hann,fs_ecg)            #veo la respuesta
rf = 20*np.log10(h)
plt.figure (20)
plt.plot(w,rf)

##fir hann 2001
##fir FIRLS 2001

ECG_f_butt = scps.sosfilt(sos_butt, ecg_one_lead) 
ECG_f_cheb2 = scps.sosfilt(sos_cheby2, ecg_one_lead) 
ECG_f_cauer = scps.sosfilt(sos_cauer, ecg_one_lead)

regs_interes = (
        [4000, 5500], # muestras
        [10e3, 11e3], # muestras
        )

for ii in regs_interes:
   
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
   
    plt.figure()
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    plt.plot(zoom_region, ECG_f_butt[zoom_region], label='Butterworth')
    plt.plot(zoom_region, ECG_f_cheb2[zoom_region], label='cheb2')
    plt.plot(zoom_region, ECG_f_cauer[zoom_region], label='cauer')
    #plt.plot(zoom_region, ECG_f_hann[zoom_region], label='hann')
    #plt.plot(zoom_region, ECG_f_hamming[zoom_region], label='hamming')
    #plt.plot(zoom_region, ECG_f_boxcar[zoom_region], label='boxcar')
    
    plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
   
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
           
    plt.show()


