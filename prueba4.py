import yaml
from PIL import Image
import cv2
import numpy as np

class Diccionario:
    with open('YOLO_PokerCards_Vision.yaml', 'r') as file:
        data = yaml.safe_load(file)

    cartas = {}
    for key, value in data['names'].items():
        carta_nombre = value
        carta_ruta = f'Cartas/{value.lower()}.jpeg'
        cartas[carta_nombre] = Image.open(carta_ruta)
        cartas[carta_nombre] = cartas[carta_nombre].resize((cartas[carta_nombre].size[0] // 6, cartas[carta_nombre].size[1] // 6))
        
    

miDiccionario = Diccionario()


cv2.imshow("Carta", cv2.cvtColor(np.array(miDiccionario.cartas['10 Corazones']), cv2.COLOR_RGB2BGR))
cv2.waitKey(0)