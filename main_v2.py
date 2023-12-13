#Librerias
import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import yaml

#Definicion de argumentos
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO_PokerCards_Vision")
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--predict", action='store_true')
    parser.add_argument("--valid", action='store_true')
    parser.add_argument("--webcam_resolution",
        default = [1280, 720],
        nargs = 2,
        type = int
    )
    args = parser.parse_args()
    return args

def NumeroJugadores():
    players = None
    try:
        players = int(input("\nDime el numero de jugadores (minimo 2, maximo 3): "))
        print("El numero introducido es:", players)
        if players != 2 and players != 3:
            print("El numero de jugadores introducido no es correcto")
            players = NumeroJugadores()
    except:
        print("El dato introducido no es correcto, introduce un numero REAL")
        players = NumeroJugadores()
    return players

def SeleccionCamara(playerName):
    camara_index = 0
    while True:
        cap = cv2.VideoCapture(camara_index)

        if not cap.isOpened():
            break

        cap.release()
        camara_index += 1

    camara = []
    for cam in range(camara_index):
        camara.append(cv2.VideoCapture(cam))

    print(f"\nVisualice que camara quiere utilizar para {playerName} (Presione ESC, para salir de visualizador):")

    while True:
        for index in range(camara_index):
            _, frame = camara[index].read()
            cv2.imshow("Camara " + str(index), frame)
    
        if (cv2.waitKey(30) == 27): #Si se presiona ESC, se sale de la ventana
            break
    cv2.destroyAllWindows()
    
    CamaraFinal = None
    while CamaraFinal != int:
        try:   
            CamaraFinal = int(input(f"\nCual es el INDEX de la camara de {playerName}: "))
            if CamaraFinal <= camara_index and CamaraFinal >= 0:
                print(f"La camara elejida para {playerName} es: ", CamaraFinal)
                break
            print("La camara elegida esta fuera del rango de camaras seleccionables")
            print("Por favor, seleccione una camara dentro del rango")
        except:
            print("El dato introducido no es correcto, introduce un numero REAL")

    return CamaraFinal
    
class Jugador:
    NumeroJugadores = 0
    def __init__(self, video_param, croupier = None):
        Jugador.NumeroJugadores += 1
        if croupier == True:
            self.name = "Croupier"
            self.points = [(645,100), (775,100), (905,100), (1035,100), (1165,100)]
            self.textPoint = ((645,300))
        else:
            self.name = input("\nDime el nombre del jugador: ")
            if Jugador.NumeroJugadores == 2:
                self.points = [(145,500), (275,500)]
                self.textPoint = ((145,450))
            if Jugador.NumeroJugadores == 3:
                self.points = [(1545,500), (1675,500)]
                self.textPoint = ((1545,450))
            if Jugador.NumeroJugadores == 4:
                self.points = [(840,700), (965,700)]
                self.textPoint = ((840,650))
            

        self.video = cv2.VideoCapture(SeleccionCamara(self.name))
        self.model = YOLO("YOLO_PokerCards_Vision_Final.pt") #Modelo YOLOv8 de predición entrenado

        frame_width, frame_height = video_param
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    
    def leer_frame(self):
        _, frame = self.video.read()
        return frame
    
    def result(self, frame):
        return self.model(frame, agnostic_nms = True)[0]

class Diccionario:
    with open('YOLO_PokerCards_Vision.yaml', 'r') as file:
        data = yaml.safe_load(file)

    cartas = {}
    for key, value in data['names'].items():
        carta_nombre = value
        carta_ruta = f'Cartas/{value.lower()}.jpeg'
        cartas[carta_nombre] = Image.open(carta_ruta)
        cartas[carta_nombre] = cartas[carta_nombre].resize((cartas[carta_nombre].size[0] // 6, cartas[carta_nombre].size[1] // 6))


#Programa Principal
def main(args):
    #Definicion de tarea "train" (entrenamiento)
    if args.train:
        model = YOLO("YOLO_PokerCards_Vision_Final.pt") 
        results = model.train(data='YOLO_PokerCards_Vision.yaml', epochs=20, imgsz=640)

    #Definicion de tarea "valid" (validacion)
    elif args.valid:
        model = YOLO('YOLO_PokerCards_Vision.yaml')  # load an official model
        # Validate the modelz
        metrics = model.val()  # no arguments needed, dataset and settings remembered
        metrics.box.map    # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps   # a list contains map50-95 of each category

    #Definicion de tarea "predict" (prediccion) (PREDETERMINADO)
    else:
        players = []
        players.append(Jugador(args.webcam_resolution, croupier=True))
        num_players = NumeroJugadores()
        for _ in range(num_players):
            players.append(Jugador(args.webcam_resolution))
        
    miDiccionario = Diccionario()
    PokerTable = Image.open('PokerTable.jpg')
    cv2.namedWindow('Mesa', cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Mesa", args.webcam_resolution)

    while True:
        CopyPokerTable = PokerTable.copy()

        for num_players in range(Jugador.NumeroJugadores):
            frame = players[num_players].leer_frame()
            result = players[num_players].result(frame)
            detections = sv.Detections.from_ultralytics(result)
            draw = ImageDraw.Draw(CopyPokerTable)
            draw.text(players[num_players].textPoint, players[num_players].name, font=ImageFont.truetype("calibrib.ttf", 50), fill=(0, 0, 0))
            
            for i, class_id in enumerate(list(set(detections.class_id))):
                try:
                    CopyPokerTable.paste(miDiccionario.cartas[players[num_players].model.model.names[class_id]], players[num_players].points[i])
                except:
                    print("MISSED")
                    pass
            
        cv2.imshow("Mesa", cv2.cvtColor(np.array(CopyPokerTable), cv2.COLOR_RGB2BGR))

        if (cv2.waitKey(30) == 27): break #Si se presiona ESC, se sale de la ventana

#Init definition
if __name__ == "__main__":
    args = parse_arguments()
    main(parse_arguments())