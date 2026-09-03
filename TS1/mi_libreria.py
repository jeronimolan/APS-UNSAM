#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 20:07:57 2026

@author: jeronimo
"""

import numpy as np
import scipy.signal as ss
import scipy.fft as scfft

#hecha para TS0
def mi_funcion_seno(vmax, dc, ff, ph, nn, fs):
    """
    Variables de entrada
    vmax = Amplitud
    dc = componente de continua
    ff = frecuencia de señal
    ph = desplazamiento de fase
    nn = numero de muestras
    fs = frecuencia de muestreo
    Salida
    dt, amplitud a dt 
    """
    
    tt =  np.arange(start = 0, stop = nn/fs, step = 1/fs)
    xx = dc + vmax * np.sin(2*np.pi*ff*tt + ph)
    return tt, xx

#hecha para TS0
def mi_funcion_dienteDeSierra(vmax, dc, width, ff, nn, fs):
    """
    Variables de entrada
    vmax = Amplitud
    dc = componente de continua
    width = ancho de pendiente de subida
    ff = frecuencia de señal
    nn = numero de muestras
    fs = frecuencia de muestreo
    Salida
    dt, amplitud a dt 
    """
    tt =  np.arange(start = 0, stop = nn/fs, step = 1/fs)
    xx = dc + vmax * ss.sawtooth( 2*np.pi*tt*ff , width)
    return tt, xx

#hecha para TS1
def mi_ruido(A,dist,dc,n,fs):
    """
    A = amplitud de la señal
    dc = componente de continua
    dist = distribucion (normal, uniform)
    n = numero de muestras
    """
    potencia = (A**2)/2
    noise = []
    if dist == "normal":    
        for i in range(n):
            noise.append(np.random.normal(0,np.sqrt(potencia)) + dc)
    elif dist == "uniform":
        for i in range(n):
            noise.append(np.random.uniform(-A,A) + dc)
            #La potencia media de una distribucion uniforme es (vmax - vmin)²/12
            #despejando, siendo que vmax = -vmin, queda que P = 4vmax²/12
            #simplificando, P = vmax²/3
            #la vmax me da una amplitud, siendo que la distribucion la centro en dc,
            #entonces vmax = A = sqrt(P*3)
    else:
        print("Distribucion no disponible")
        return noise
    
    tt =  np.arange(start = 0, stop = n/fs, step = 1/fs)
    return tt, noise
    
#hecha para TS1
def mi_cuadrada(A,duty,dc,ff,n,fs):
    tt =  np.arange(start = 0, stop = n/fs, step = 1/fs)
    xx = A * ss.waveforms.square(2*np.pi*ff*tt,duty) + dc
    return tt,xx

#hecha en clase
def mi_seno_SNR(vmax, dc, ff, ph, n, fs, SNR):
    """
    Variables de entrada
    vmax = Amplitud
    dc = componente de continua
    ff = frecuencia de señal
    ph = desplazamiento de fase
    nn = numero de muestras
    fs = frecuencia de muestreo
    SNR = Relacion señal ruido
    Salida
    dt, amplitud a dt 
    """
    
    tt =  np.arange(start = 0, stop = n/fs, step = 1/fs)
    xx = dc + vmax * np.sin(2*np.pi*ff*tt + ph)
    potencia_seno  = (1/n)*np.sum(np.square(xx))
    potencia_ruido = potencia_seno * 10**(-SNR/10)
    noise = []
    for i in range(n):
        noise.append(np.random.normal(0,np.sqrt(potencia_ruido)))
        
    xx_snr = xx + noise
    return tt, xx_snr

#hecha en clase
def seno_con_ruido_discreto (A, k , Af, SNR, n, fs):
    tt = np.arange(start = 0, stop = fs/n, step = 1/fs)

    #señal y dominio de w
    xx = A*np.sin(Af * k * (fs/n) * tt)
    
    potencia_seno  = (1/n)*np.sum(np.square(xx))
    potencia_ruido = potencia_seno * 10**(-SNR/10)
    
    noise = []
    for i in range(n):
        noise.append(np.random.normal(0,np.sqrt(potencia_ruido)))
    xx_snr = xx + noise
    return tt, xx_snr

#hecha en clase
def digitalizador (signal,vfs,B,n,fs):
    tt = np.arange(start = 0, stop = fs/n, step = 1/fs)
    q = 2* vfs / 2**B
    xx =  q * np.round(signal//q)
    return tt,xx

#hecha en clase
def spectometer (signal,n,fs):
    ww = np.arange(start = 0, stop = fs/2, step = fs/n)
    #transformo la señal
    trafo_signal_N = scfft.fft(signal,n)
    trafo_signal = 1/n * trafo_signal_N[:n//2]
    #doy modulo y fase
    modulo_trafo_signal = np.abs(trafo_signal)
    fase_trafo_signal = np.angle(trafo_signal)
    modulo_db = 20 * np.log10(2*modulo_trafo_signal)
    return ww,modulo_db,fase_trafo_signal