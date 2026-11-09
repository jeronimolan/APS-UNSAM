#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 20:36:38 2026

@author: jeronimo
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scps

fs = 1000
ws1 = 0.1 
wp1 = 1 
ws2 = 40 
wp2 = 35 

log_wp1 = np.log10(wp1)

log_wp2 = np.log10(wp2)
log_ws2 = np.log10(ws2)

w1 = np.logspace(0,log_wp1,num = 100)
w2 = np.logspace(log_wp2,log_ws2,num = 100)
wpass = np.linspace(1,35,100)

atenuacion = 40.0 #db

ganancias = np.array([-atenuacion, -atenuacion, 0.0, 0.0, -atenuacion, -atenuacion])

ganancias_v = 10 ** ( ganancias / 20 )

taps_win = 1501

b_win_hann = scps.firwin2(taps_win, [0.0, ws1, wp1 ,wp2, ws2, fs/2], ganancias_v, fs = fs, window= 'hann') 

w_muestras = np.concatenate([w1,wpass,w2]) * 2* np.pi / fs

w,h = scps.freqz(b_win_hann,fs)            #veo la respuesta
rf = 20*np.log10(np.abs(h))
plt.plot(w,rf)

plt.figure(2)
w,h = scps.freqz(b_win_hann,fs,worN=w_muestras)            #veo la respuesta
rf = 20*np.log10(np.abs(h))
plt.plot(w_muestras,rf)









