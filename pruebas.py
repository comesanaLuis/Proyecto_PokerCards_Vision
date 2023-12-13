import cv2

def Mesa(Jugador):
    while True:
        for i in range(Jugador.NumeroJugadores):
            _, frame = Jugador[i].video.read()
            result = Jugador[i](frame)[0]
            detections = sv.Detections.from_ultralytics(result) #Se extraen a variables atributos de cada objeto detectado

        if (cv2.waitKey(30) == 27) :#Si se presiona ESC, se sale de la ventana
            break



if __name__ == "__main__":
    Mesa()
