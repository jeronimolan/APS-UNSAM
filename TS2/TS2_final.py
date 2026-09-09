#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 16:56:33 2026

@author: jeronimo
"""

import numpy as np
import matplotlib.pyplot as plt
import mi_libreria as lib

"""
En esta tarea semanal retomamos la consigna de la tarea anterior, donde 
simulamos el bloque de cuantización de un ADC de B bits en un rango de  
±VF Volts. Ahora vamos a completar la simulación del ADC incluyendo la 
capacidad de muestrear a fs Hertz.

Para ello se simulará el comportamiento del dispositivo al 
digitalizar una senoidal contaminada con un nivel predeterminado de 
ruido. Comenzaremos describiendo los parámetros a ajustar de la senoidal:

    frecuencia f0 arbitraria, por ejemplo f0=fS/N=Δf 
    energía normalizada, es decir energía (o varianza) unitaria

Con respecto a los parámetros de la secuencia de ruido, diremos que:

    será de carácter aditivo, es decir la señal que entra al ADC será sR=s+n. 
    Siendo n la secuencia que simula la interferencia, y s la senoidal 
    descrita anteriormente.
    La potencia del ruido será Pn=kn.Pq W siendo el factor k una escala 
    para la potencia del ruido de cuantización Pq=q**2/12.
    finalmente, n será incorrelado y Gaussiano.

El ADC que deseamos simular trabajará a una frecuencia de muestreo fS=1000 Hz 
y tendrá un rango analógico de ±VF=2 Volts.

Se pide:

a) Generar el siguiente resultado producto de la experimentación. B = 4 bits, 
    kn=1.
b) Analizar para una de las siguientes configuraciones B = ̣{4, 8 y 16} bits, 
    kn={1/10,1,10}. Discutir los resultados respecto a lo obtenido en a).
"""

fs = 1000
N = 1000

f0 = fs/N

B = 4
Vfs = 2
q = (Vfs*2)/(2**B)
kn = 1
Pq = (q**2)/12
Pn = kn * Pq
A_seno = np.sqrt(2)
#Sr = s + n

tt,n = lib.mi_ruido(Pn, "normal", 0, N, fs)

tt,s = lib.mi_funcion_seno(A_seno, 0, f0, 0, N, fs)

Sr = s + n

tt,Sq = lib.digitalizador(Sr, q, N, fs)

plt.figure(1)
plt.title(f"Señal senoidal con ruido, muestreada por un ADC de 4 bits - ±Vf = 2.0V - q = {q:.2f} V")
plt.plot(tt,Sr, ':*')
plt.plot(tt,Sq)
plt.plot(tt,s,'--')
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.legend(("Sr (ADC IN)","Sq (ADC OUT)","S (Seno)"))


plt.figure(2)
plt.title(f"Señal senoidal con ruido, muestreada por un ADC de 4 bits - ±Vf = 2.0V - q = {q:.2f} V")
ww,Sq_g,Sq_f = lib.spectometer(Sq, N, fs, A_seno)

ww,n_g,n_f = lib.spectometer(n, N, fs, A_seno)

nq = Sr - Sq

ww,nq_g,nq_f = lib.spectometer(nq, N, fs, A_seno)

ww,n_g,n_f = lib.spectometer(n, N, fs, A_seno)

ww,Sr_g,Sr_f = lib.spectometer(Sr, N, fs, A_seno)

plt.plot(ww,Sq_g, color = 'green')
plt.plot(ww,n_g,':', color = 'red')
plt.plot(ww,nq_g,':', color = 'blue')

plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad de Potencia [dB]")
plt.legend(("Sq (ADC OUT)","n","nq"))

plt.figure(3)
plt.title("Ruido de cuantización para un ADC de 4 bits - ±Vf = 2.0V - q = 0.25V")
plt.hist(nq)
plt.plot(
    [-q/2, q/2, q/2, -q/2, -q/2],
    [0, 0, 100, 100, 0],
    linestyle=':',
    linewidth=2
)


#Para kn = 1/10, B = 8

B = 8
Vfs = 2
q = (Vfs*2)/(2**B)
kn = 10
Pq = (q**2)/12
Pn = kn * Pq
A_seno = np.sqrt(2)
#Sr = s + n

tt,n = lib.mi_ruido(Pn, "normal", 0, N, fs)

tt,s = lib.mi_funcion_seno(A_seno, 0, f0, 0, N, fs)

Sr = s + n

tt,Sq = lib.digitalizador(Sr, q, N, fs)

plt.figure(4)
plt.title(f"Señal senoidal con ruido, muestreada por un ADC de 8 bits - ±Vf = 2.0V - q = {q:.4f} V")
plt.plot(tt,Sr, ':*')
plt.plot(tt,Sq)
plt.plot(tt,s)
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.legend(("Sr (ADC IN)","Sq (ADC OUT)","S (Seno)"))


plt.figure(5)
plt.title(f"Señal senoidal con ruido, muestreada por un ADC de 8 bits - ±Vf = 2.0V - q = {q:.4f} V")
ww,Sq_g,Sq_f = lib.spectometer(Sq, N, fs, A_seno)


ww,n_g,n_f = lib.spectometer(n, N, fs, A_seno)


nq = Sr - Sq


ww,nq_g,nq_f = lib.spectometer(nq, N, fs, 2)


plt.plot(ww,Sq_g, color = 'green')
plt.plot(ww,nq_g,':', color = 'blue')
plt.plot(ww,n_g,':', color = 'red')


plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad de Potencia [dB]")
plt.legend(("Sq (ADC OUT)","nq","n"))


plt.figure(6)
plt.title(f"Ruido de cuantización para un ADC de 8 bits - ±Vf = 2.0V - q = {q:.4f} V")
plt.hist(nq)
plt.plot(
    [-q/2, q/2, q/2, -q/2, -q/2],
    [0, 0, 100, 100, 0],
    linestyle=':',
    linewidth=2
)
