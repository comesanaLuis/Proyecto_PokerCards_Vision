from PIL import Image
import cv2
import numpy as np


CartasDetectadas = ['10 Diamantes', '4 Treboles']


def MostrarCartas(ALL_class_id):
    carta = []
    for i in range(1):
        #carta.append(Image.open(model.model.names[class_id]+".jpg"))
        carta.append(Image.open('Cartas/' + ALL_class_id[i] +".jpeg"))
        carta[i].resize((carta.size[i]//6, carta.size[i]//6))
        
point = [(645,100), (775,100), (905,100), (1035,100), (1165,100), (145,500), (275,500), (1545,500), (1675,500)]
PokerTable = Image.open('PokerTable.jpg')


cv2.namedWindow('Mesa', cv2.WINDOW_NORMAL)
cv2.resizeWindow("Mesa", (1280, 720))

while True:
    CopyPokerTable = PokerTable.copy()
    CartasCargadas = MostrarCartas(CartasDetectadas)

    for i in range(CartasCargadas):
        CopyPokerTable.paste(CartasCargadas[i], point[i])

    cv2.imshow("Mesa", cv2.cvtColor(np.array(CopyPokerTable), cv2.COLOR_RGB2BGR))

    if (cv2.waitKey(30) == 27): break #Si se presiona ESC, se sale de la ventana
        
