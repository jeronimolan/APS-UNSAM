#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 01:56:01 2026

@author: jeronimo
"""

"""
En este primer trabajo comenzaremos por diseñar un generador de señales que utilizaremos en las primeras simulaciones que hagamos. La primer tarea consistirá en programar una función que genere señales senoidales y que permita parametrizar:

    la amplitud máxima de la senoidal (volts)
    su valor medio (volts)
    la frecuencia (Hz)
    la fase (radianes)
    la cantidad de muestras digitalizada por el ADC (# muestras)
    la frecuencia de muestreo del ADC.

es decir que la función que uds armen debería admitir se llamada de la siguiente manera

tt, xx = mi_funcion_sen( vmax = 1, dc = 0, ff = 1, ph=0, nn = N, fs = fs)
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss

fs = 1000#Hz
N = 1000#muestras

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

x,y = mi_funcion_seno(vmax = 1, dc = 0, ff = 1, ph = 0, nn = N, fs = fs)
plt.figure(1)
plt.title("1Hz")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (V)")
plt.plot(x,y)

"""
Bonus: ff = 500Hz, 999Hz, 1001Hz, 2001Hz
"""

x,y = mi_funcion_seno(vmax = 1, dc = 0, ff = 500, ph = 0, nn = N, fs = fs)
plt.figure(2)
plt.title("500Hz")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (V)")
plt.plot(x,y)


x,y = mi_funcion_seno(vmax = 1, dc = 0, ff = 999, ph = 0, nn = N, fs = fs)
plt.figure(3)
plt.title("999Hz")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (V)")
plt.plot(x,y)


x,y = mi_funcion_seno(vmax = 1, dc = 0, ff = 1001, ph = 0, nn = N, fs = fs)
plt.figure(4)
plt.title("1001Hz")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (V)")
plt.plot(x,y)


x,y = mi_funcion_seno(vmax = 1, dc = 0, ff = 2001, ph = 0, nn = N, fs = fs)
plt.figure(5)
plt.title("2001Hz")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (V)")
plt.plot(x,y)


"""
Bonus: Implementacion de diente de sierra
"""

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

x,y = mi_funcion_dienteDeSierra(vmax = 1, dc = 0, width = 0.2, ff = 10, nn = N, fs = fs)
plt.figure(6)
plt.title("1Hz")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (V)")
plt.plot(x,y)


