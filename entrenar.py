import cv2
import os
import numpy as np

def entrenar():
    direccion = "D:/fotos/rostro_sin_tapabocas"
    lista = os.listdir(direccion)

    etiquetas = []
    rostros = []
    con = 0

    for nameDir in lista:
        nombre = direccion + '/' + nameDir

        for fileName in os.listdir(direccion):
            etiquetas.append(con)
            rostros.append(cv2.imread(nombre + '/' + fileName, 0))

        con += 1
        print("Generado para ", con)

    print("Entrenando modelo")
    reconocimiento = cv2.face.LBPHFaceRecognizer_create()
    reconocimiento.train(rostros, np.array(etiquetas))
    reconocimiento.write('modeloLBP.xml')
    print("modelo creado")
