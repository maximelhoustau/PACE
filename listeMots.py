#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 15:15:37 2018

@author: eburle
"""

import RPi.GPIO as GPIO                                               # Importation des librairies qui gerent les ports
import time                                                           # Importation de la librairie temps

GPIO.setmode(GPIO.BCM)                                                # BCM : Numero des GPIO (GPIO 18)
GPIO.setup(18, GPIO.OUT)                                              # Definition du port en sortie
GPIO.setwarnings(False)                                               # Mettre sur OFF les alertes (qui sont inutiles)


# Affichage de texte
print("\n+------------------/ Blink LED /------------------+")
print("|                                                 |")
print("| La LED doit etre reliee au GPIO 18 du Raspberry |")
print("|                                                 |")
print("+-------------------------------------------------+\n")

nbrBlink = input("Combien de fois la LED doit clignoter ?\n")          # Utilisation de la fonction input pour acquerir des informations
tempsAllume = input("Combien de temps doit-elle rester allumee ?\n")
tempsEteint = input("Combien de temps doit-elle rester eteinte ?\n")

i = 0                                                                  # Definition d'une variable type compteur

while i < nbrBlink :
    GPIO.output(18, True)                                              # Mise a 1 du GPIO 18 (+5V)
    time.sleep(tempsAllume)                                            # On attend le temps defini
    GPIO.output(18, False)                                             # Mise a zero du GPIO 18 (GND)
    time.sleep(tempsEteint)                                            # ...
    i = i+1

GPIO.cleanup()

G = ([[1, 0, 0, 0,1,0,1],
             [0, 1, 0, 0,1,1,0],
             [0, 0, 1, 0,1,1,1],
             [0, 0, 0, 1,0,1,1]])

def mots(G):
    i = len(G)
    j = len(G[0])
    L = [[0]*j]
    l = [0]*i
    for k in range(0,j):
        m1 = [0]*j
        m1[k] = 1
        L.append(m1)
    for k in range(1, 2**i):
        m = [0]*j
        p = 0
        while (l[p] == 1):
            l[p] = 0
            p += 1
        l[p] = 1
        for u in range(0,i):
            if (l[u] == 1):
                m = [(m[t] + G[u][t])%2 for t in range (0,j)]
        L.append(m)
        for t in range (0,j):
            m1 = list(m)
            m1[t] = (m1[t] + 1)%2
            L.append(m1)
    return L