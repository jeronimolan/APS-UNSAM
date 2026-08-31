#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#el codigo se escribio con asistencia de chat-gpt para el formateo de los gráficos

"""
Created on Thu Aug 27 13:59:43 2026

@author: jeronimo
"""
"""
Ejerc""icio 1:  

Utilizando siempre N = 1000 muestras. Se pide:

Sintetizar:

    1-Señal sinusoidal de 2 KHz que tenga al menos 10 puntos por período.
    2-Misma señal con 2 W de potencia media y desfasada en π/2.
    3-Una secuencia aleatoria de ruido normalmente distribuido con DC (valor medio) 0V y varianza 0.1 W.
    4-Una secuencia aleatoria de ruido uniformemente distribuido con DC (valor medio) 0V y varianza 0.1 W. 
    5-Un pulso rectangular de la misma frecuencia, 1 W de potencia y ciclo de actividad del 50% (Ver scipy.signal apartado Waveforms).

Para cada señal visualice el módulo de la transformada de Fourier.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
import mi_libreria as lib #mi libreria con funciones hechas durante la cursada


N = 1000

"1"
#10 puntos por periodo = 2khz*10 = fs = 20000
fs = 20000
t1,x1 = lib.mi_funcion_seno(vmax = 1,dc = 0, ff = 2000,ph = 0,nn = N,fs = fs)
#transformada
ww,modulo_tf1,fase_tf1 = lib.spectometer(x1, N, fs)

fig, (fdt,fdf) = plt.subplots(2,1)
fig.suptitle("Señal sinusoidal de 2 KHz que tenga al menos 10 puntos por período")
fig.tight_layout(h_pad = 2)
#subplot de la funcion en el tiempo
fdt.set_title("Señal")
fdt.set_xlabel("Tiempo (s)")
fdt.set_ylabel("Tension (V)")
fdt.set_xlim(0,3*1/2000)
fdt.plot(t1,x1,':o')
#fdt.xlim(0, 3*1/2000) #muestro 3 ciclos de la seno
#subplot en la frecuencia
fdf.set_title("Espectro")
fdf.set_xlabel("Frecuencia (Hz)")
fdf.set_ylabel("Amplitud (dB)")
fdf.plot(ww,modulo_tf1)



"2"
fs = 20000
t2,x2 = lib.mi_funcion_seno(vmax = 2,dc = 0, ff = 2000,ph = np.pi/2,nn = N,fs = fs)
potencia_media = (1/N)*np.sum(np.square(x2)) # demuestro que tiene 2W de potencia media
#no implemente la funcion con la potencia, sino con la amplitud/tension maxima
#transformada
ww2,modulo_tf2,fase_tf2 = lib.spectometer(x2, N,fs)

fig2, (fdt2,fdf2) = plt.subplots(2,1)
fig2.suptitle(f"Misma señal con {potencia_media:.2f} W de potencia media y desfasada en π/2.")
fig2.tight_layout(h_pad = 2)
#f me permite mostrar el valor de variables en strings, como si fueran parte del string
#con .2f reduzco el tamaño del float a 2 decimales
#subplot de la funcion en el tiempo
fdt2.set_title("Señal")
fdt2.set_xlabel("Tiempo (s)")
fdt2.set_ylabel("Tensión (V)")
fdt2.set_xlim(0,3*1/2000)
fdt2.plot(t2,x2,':o')
#fdt.xlim(0, 3*1/2000) #muestro 3 ciclos de la seno
#subplot en la frecuencia
fdf2.set_title("Espectro")
fdf2.set_xlabel("Frecuencia (Hz)")
fdf2.set_ylabel("Amplitud (dB)")
fdf2.plot(ww2,modulo_tf2)


"3"
fs = 20000
t3,x3 = lib.mi_ruido(A = np.sqrt(2 * 0.1) , dist = "normal" , dc = 0 , n = N ,fs = fs)
potencia_ruido = (1/N)*np.sum(np.square(x3)) #demuestro que tiene 0.1W de potencia media
ww3,modulo_tf3,fase_tf3 = lib.spectometer(x3, N,fs)

fig3, (fdt3,fdf3) = plt.subplots(2,1)
fig3.suptitle(f"Ruido normalmente distribuido, DC = 0V, varianza {potencia_ruido:0.2f} W")
fig3.tight_layout(h_pad = 2)
#subplot de la funcion en el tiempo
fdt3.set_title("Señal")
fdt3.set_xlabel("Tiempo (s)")
fdt3.set_ylabel("Tensión (V)")
fdt3.plot(t3,x3,':o')
fdf3.set_title("Espectro")
fdf3.set_xlabel("Frecuencia (Hz)")
fdf3.set_ylabel("Amplitud (dB)")
fdf3.plot(ww3,modulo_tf3)


"4"
fs = 20000
t4,x4 = lib.mi_ruido(A = np.sqrt(3*0.1) , dist = "uniform" , dc = 0 , n = N ,fs = fs)
potencia_ruido = (1/N)*np.sum(np.square(x4)) #demuestro que tiene 0.1W de potencia media
ww4,modulo_tf4,fase_tf4 = lib.spectometer(x4, N,fs)

fig4, (fdt4,fdf4) = plt.subplots(2,1)
fig4.suptitle(f"Ruido uniformemente distribuido, DC = 0V, varianza {potencia_ruido:0.2f} W")
fig4.tight_layout(h_pad = 2)
#subplot de la funcion en el tiempo
fdt4.set_title("Señal")
fdt4.set_xlabel("Tiempo (s)")
fdt4.set_ylabel("Tensión (V)")
fdt4.plot(t4,x4,':o')
fdf4.set_title("Espectro")
fdf4.set_xlabel("Frecuencia (Hz)")
fdf4.set_ylabel("Amplitud (dB)")
fdf4.plot(ww4,modulo_tf4)


"5"
fs = 20000
ff = 2000
t5,x5 = lib.mi_cuadrada(A = np.sqrt(2) , duty= 0.5 , dc= 0, ff= 2000 , n = N ,fs = fs)
potencia_ruido = (1/N)*np.sum(np.square(x5)) #demuestro que tiene 1 de potencia media
ww5,modulo_tf5,fase_tf5 = lib.spectometer(x5, N,fs)

fig5, (fdt5,fdf5) = plt.subplots(2,1)
fig5.suptitle(f"Un pulso rectangular igual f, duty 50%, potencia {potencia_ruido:0.2f} W")
fig5.tight_layout(h_pad = 2)
#subplot de la funcion en el tiempo
fdt5.set_title("Señal")
fdt5.set_xlabel("Tiempo (s)")
fdt5.set_ylabel("Tensión (V)")
fdt5.set_xlim(0,3*1/2000)
fdt5.plot(t5,x5,':o')
fdf5.set_title("Espectro")
fdf5.set_xlabel("Frecuencia (Hz)")
fdf5.set_ylabel("Amplitud (dB)")
fdf5.plot(ww5,modulo_tf5)


"""
Bonus
    -Implementar alguna otra señal disponible en scipy.signal
    -Averiguar como se podria medir la potencia mediante la transformada de fourier 
    (teorema de parseval)
"""
#implemento un pulso

def mi_pulso(pos,n,fs):
    tt = np.arange(start = 0, stop = n/fs, step = 1/fs)
    pulso = ss.unit_impulse(64,pos)
    xx = []
    for i in range(n):
        if i < 64: xx.append(pulso[i])
        else: xx.append(0)
    return tt, xx

#pruebo el pulso
fs = 20000
ff = 2000
t6,x6 = mi_pulso(1, N, fs)
ww6,modulo_tf6,fase_tf6 = lib.spectometer(x6, N,fs)

fig6, (fdt6,fdf6) = plt.subplots(2,1)
fig6.suptitle("Implementacion de funcion pulso")
fig6.tight_layout(h_pad = 2)
#subplot de la funcion en el tiempo
fdt6.set_title("Señal")
fdt6.set_xlabel("Tiempo (s)")
fdt6.set_ylabel("Tensión (V)")
fdt6.plot(t6,x6,':o')
fdf6.set_title("Espectro")
fdf6.set_xlabel("Frecuencia (Hz)")
fdf6.set_ylabel("Amplitud (dB)")
fdf6.plot(ww6,modulo_tf6)

#por la identidad de parseval, para una dft, la potencia de una señal se 
#puede medir como

import scipy.fft as scfft
t,x = lib.mi_funcion_seno(np.sqrt(2),0,2000,0,N,fs)
potencia = (1/N**2)*np.sum(np.abs(scfft.fft(x,N))**2)
print(potencia)









