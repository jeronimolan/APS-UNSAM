#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 17:15:07 2026

@author: jeronimo
"""

import numpy as np
import matplotlib.pyplot as plt
import mi_libreria as lib

"""
Luego, haremos el siguiente experimento:

    Senoidal de frecuencia f0=k0∗fS/N=k0.Δf
    potencia normalizada, es decir energía (o varianza) unitaria

Se pide:

a) Sea k0 

    N/4 
    N/4+0.25 
    N/4+0.5 

Notar que a cada senoidal se le agrega una pequeña desintonía respecto a  Δf. 
Graficar las tres densidades espectrales de potencia (PDS's) y discutir cuál es el 
efecto de dicha desintonía en el espectro visualizado.

b) Verificar la potencia unitaria de cada PSD, puede usar la identidad de Parseval. 
En base a la teoría estudiada. Discuta la razón por la cual una señal senoidal tiene un 
espectro tan diferente respecto a otra de muy pocos Hertz de diferencia. 

c) Repetir el experimento mediante la técnica de zero padding. 
Dicha técnica consiste en agregar ceros al final de la señal para aumentar Δf de forma 
ficticia. Probar agregando un vector de 9*N ceros al final. Discuta los resultados obtenidos.
"""

N = 1000
fs = 1000

#a)
tt,x1 = lib.seno_con_k(np.sqrt(2), N/4, fs/N, N, fs)
ww,spec_t1,spec_t1f = lib.spectometer(x1, N, fs, np.sqrt(2))

plt.figure(1)
plt.title("Señal Senoidal con k0 = N/4 (250Hz)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad de potencia espectral [dB/Hz]")
plt.plot(ww,spec_t1, ":x")

tt,x2 = lib.seno_con_k(np.sqrt(2), N/4 + 0.25, fs/N, N, fs)
ww,spec_t2,spec_t2f = lib.spectometer(x2, N, fs, np.sqrt(2))

plt.figure(2)
plt.title("Señal Senoidal con k0 = N/4 + 0.25 (250.25Hz)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad de potencia espectral [dB/Hz]")
plt.plot(ww,spec_t2, ":x")

tt,x3 = lib.seno_con_k(np.sqrt(2), N/4 + 0.5, fs/N, N, fs)
ww,spec_t3,spec_t3f = lib.spectometer(x3, N, fs, np.sqrt(2))

plt.figure(3)
plt.title("Señal Senoidal con k0 = N/4 (250+0.5Hz)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad de potencia espectral [dB/Hz]")
plt.plot(ww,spec_t3, ":x")

#el efecto es subir el piso de densidad de potencia (graficamente)


#b)
#Parseval de TS1

import scipy.fft as scfft
potencia1 = (1/N**2)*np.sum(np.abs(scfft.fft(x1,N))**2)
print(potencia1)
potencia2 = (1/N**2)*np.sum(np.abs(scfft.fft(x2,N))**2)
print(potencia2)
potencia3 = (1/N**2)*np.sum(np.abs(scfft.fft(x3,N))**2)
print(potencia3)


#lo que sucede es que se toman muestras desplazadas de aquellas con una relacion
#k entero veces con el muestreo. Ese pequeño desplazamiento hace que se tomen muestras
#en los lobulos de la sinc que corresponde a la transformada de laventana implicita que 
# encierra a nuestras seccion de seno

#c)

x1_padding = np.zeros(10*N)
x1_padding[0:N] = x1
ww1,spec_t1p,spec_t1pf = lib.spectometer(x1_padding, 10*N, fs, np.sqrt(2))
plt.figure(4)
plt.title("Señal Senoidal con k0 = N/4 (250Hz)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad de potencia espectral [dB/Hz]")
plt.plot(ww1,spec_t1p, ":x")


x2_padding = np.zeros(10*N)
x2_padding[0:N] = x2
ww2,spec_t2p,spec_t2pf = lib.spectometer(x2_padding, 10*N, fs, np.sqrt(2))
plt.figure(5)
plt.title("Señal Senoidal con k0 = N/4 (250+0.25Hz)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad de potencia espectral [dB/Hz]")
plt.plot(ww2,spec_t2p, ":x")

x3_padding = np.zeros(10*N)
x3_padding[0:N] = x3
ww3,spec_t3p,spec_t3pf = lib.spectometer(x3_padding, 10*N, fs, np.sqrt(2))
plt.figure(6)
plt.title("Señal Senoidal con k0 = N/4 (250+0.5Hz)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad de potencia espectral [dB/Hz]")
plt.plot(ww3,spec_t3p, ":x")







