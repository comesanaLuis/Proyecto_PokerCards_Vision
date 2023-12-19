#Librerias
import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import yaml
import pied_poker as pp
from multiprocessing import Process, Pipe

#Definicion de los argumentos principales
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

#Definicion de la funcion que pregunta el numero de jugadores de la partida
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

#Definicion de la funcion que lista las camaras y se asigna una para asociarla a un jugador
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

#Definicion de la clase Jugador, en ella se recogen las variables principales como nombre del jugador, camara asociada,
#puntos donde visualizar las cartas y texto en la mesa y modelo de prediccion
class Jugador:
    NumeroJugadores = 0
    def __init__(self, video_param, croupier = None):
        Jugador.NumeroJugadores += 1
        if croupier == True:
            self.name = "Croupier"
            self.points = [(645,100), (775,100), (905,100), (1035,100), (1165,100)]
            self.textNamePoint = ((645,300))
        else:
            self.name = input("\nDime el nombre del jugador: ")
            if Jugador.NumeroJugadores == 2:
                self.points = [(145,500), (275,500)]
                self.textNamePoint = ((145,450))
                self.textPoint = ((145,300))
            if Jugador.NumeroJugadores == 3:
                self.points = [(1545,500), (1675,500)]
                self.textNamePoint = ((1545,450))
                self.textPoint = ((1450,300))
            if Jugador.NumeroJugadores == 4:
                self.points = [(740,700), (865,700)]
                self.textNamePoint = ((740,650))
                self.textPoint = ((740,500))

        self.video = cv2.VideoCapture(SeleccionCamara(self.name))
        self.model = YOLO("YOLO_PokerCards_Vision_Final.pt")

        frame_width, frame_height = video_param
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    
    def leer_frame(self):
        _, frame = self.video.read()
        return frame
    
    def result(self, frame):
        return self.model(frame, agnostic_nms = True)[0]

#Definicion de la clase diccionario, su funcion es precargar todas las imagenes de las cartas de la baraja 
#para luego mostar la carta en la mesa cuando un modelo detecta una carta
class Diccionario:
    with open('YOLO_PokerCards_Vision.yaml', 'r') as file:
        data = yaml.safe_load(file)

    cartas = {}
    for key, value in data['names'].items():
        carta_nombre = value
        carta_ruta = f'Cartas/{value.lower()}.jpeg'
        cartas[carta_nombre] = Image.open(carta_ruta)
        cartas[carta_nombre] = cartas[carta_nombre].resize((cartas[carta_nombre].size[0] // 6, cartas[carta_nombre].size[1] // 6))

#Definicion de la funcion que traduce la variable de entrada correspondiente a una etiqueta del modelo de predicion a otra etiqueta 
#con otro nombre para posteriormente calcular en otra funcion la mano mas fuete y la probabilidad de ganar a futuro
def traduccion(cartas):
    diccionario_poker = {
        '10 Trevoles': '10C', '10 Diamantes': '10D', '10 Corazones': '10H', '10 Picas': '10S',
        '2 Trevoles': '2C', '2 Diamantes': '2D', '2 Corazones': '2H', '2 Picas': '2S',
        '3 Trevoles': '3C', '3 Diamantes': '3D', '3 Corazones': '3H', '3 Picas': '3S',
        '4 Trevoles': '4C', '4 Diamantes': '4D', '4 Corazones': '4H', '4 Picas': '4S',
        '5 Trevoles': '5C', '5 Diamantes': '5D', '5 Corazones': '5H', '5 Picas': '5S',
        '6 Trevoles': '6C', '6 Diamantes': '6D', '6 Corazones': '6H', '6 Picas': '6S',
        '7 Trevoles': '7C', '7 Diamantes': '7D', '7 Corazones': '7H', '7 Picas': '7S',
        '8 Trevoles': '8C', '8 Diamantes': '8D', '8 Corazones': '8H', '8 Picas': '8S',
        '9 Trevoles': '9C', '9 Diamantes': '9D', '9 Corazones': '9H', '9 Picas': '9S',
        'As Trevoles': 'AC', 'As Diamantes': 'AD', 'As Corazones': 'AH', 'As Picas': 'AS',
        'J Trevoles': 'JC', 'J Diamantes': 'JD', 'J Corazones': 'JH', 'J Picas': 'JS',
        'K Trevoles': 'KC', 'K Diamantes': 'KD', 'K Corazones': 'KH', 'K Picas': 'KS',
        'Q Trevoles': 'QC', 'Q Diamantes': 'QD', 'Q Corazones': 'QH', 'Q Picas': 'QS'
    }
    cartas_traducidas = [diccionario_poker[carta] for carta in cartas]
    return cartas_traducidas

#Definicion de la funcion que calcula la mano mas fuete actual de los jugadores de la mesa
def manoGanadora(ArrayCartasJugadores, CartasComunitarias):
    #Definicion del diccionario manos para traducir de ingles a castellano las manos ganadoras
    diccionario_manos = {
        "HighCard": "Carta Alta",
        "OnePair": "Una Pareja",
        "TwoPair": "Doble Pareja",
        "ThreeOfAKind": "Trio",
        "Straight": "Escalera",
        "Flush": "Color",
        "FullHouse": "Full (Tres de un tipo y un par)",
        "FourOfAKind": "Póker (Cuatro de un tipo)",
        "StraightFlush": "Escalera de Color",
        "RoyalFlush": "Escalera Real",
    }
    players = []
    players_info = pp.PokerRoundResult(ArrayCartasJugadores, CartasComunitarias).players_ranked
    for p in players_info:
        mano = str(p.hand).split('(')
        players.append((p.name, diccionario_manos.get(mano[0]), mano[1][:-1]))
    return players

#Definicion de la funcion que calcula mediante simulacion la probabilidad de ganar a futuro de cada uno de los jugadores de la mesa
def probGanador(conn, ArrayCartasJugadores, CartasComunitarias):
    simulation_result = pp.PokerRoundSimulator(CartasComunitarias, ArrayCartasJugadores, len(ArrayCartasJugadores)).simulate(10000, n_jobs=1)

    players = sorted(simulation_result.__rounds__[0].players_ranked, key=lambda p: p.name)
    player_win_probabilities = []

    for player in players:
        p = simulation_result.probability_of(pp.probability.events.player_wins.PlayerWins(player))
        player_win_probabilities.append((player.name , f'({p.__percent_str__})'))
    
    conn.send(player_win_probabilities)
    conn.close()

#Programa Principal
def main(args):
    #Definicion de la tarea "train" (entrenamiento)
    if args.train:
        model = YOLO("YOLO_PokerCards_Vision_Final.pt") 
        results = model.train(data='YOLO_PokerCards_Vision.yaml', epochs=20, imgsz=640)

    #Definicion de la tarea "valid" (validacion)
    elif args.valid:
        model = YOLO('YOLO_PokerCards_Vision.yaml')
        metrics = model.val()
        metrics.box.map
        metrics.box.map50
        metrics.box.map75
        metrics.box.maps

    #Definicion de la tarea "predict" (prediccion) (PREDETERMINADO)
    else:
        #Inicializacion de jugadores, camaras, variables y funciones de primera ejecucion  
        players = []
        players.append(Jugador(args.webcam_resolution, croupier=True))
        num_players = NumeroJugadores()
        for _ in range(num_players):
            players.append(Jugador(args.webcam_resolution))
 
        miDiccionario = Diccionario()
        PokerTable = Image.open('PokerTable.jpg')
        cv2.namedWindow('Mesa', cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Mesa", args.webcam_resolution)
        simulacion_process = None
        prob = None
        manos = []

        #Definicion del bucle de prediccion
        while True:
            CopyPokerTable = PokerTable.copy() #Se carga una copia de la imagen de la mesa de poker original (Reinicializa una mesa vacia)
            players_result = [] #Se inicializa la variable resultados (Reinicializa(vacia) la variable)

            for num_players in range(Jugador.NumeroJugadores): #Bucle for para para predecir las cartas y mostrarlas en la mesa para cada jugador
                frame = players[num_players].leer_frame() #Se obtiene el frame en t instante de la camara del jugador
                result = players[num_players].result(frame) #Se predice las posibles cartas
                detections = sv.Detections.from_ultralytics(result) #Se extrae la informacion del resultado de prediccion
                
                draw = ImageDraw.Draw(CopyPokerTable) #Se inicializa el metodo draw sobre la copia de la mesa
                # Se escibe el nombre del jugador sobre la mesa
                draw.text(players[num_players].textNamePoint, players[num_players].name, font=ImageFont.truetype("calibrib.ttf", 50), fill=(0, 0, 0))

                #Se muestran las cartas en la posicion del jugador en la mesa
                cartas = []
                for i, class_id in enumerate(list(set(detections.class_id))): #list(set())) se desechan cartas duplicadas (sobles puntos de deteccion)
                    try:
                        #Se pega cada carta del jugador en su posicion correspondiente
                        CopyPokerTable.paste(miDiccionario.cartas[players[num_players].model.model.names[class_id]], players[num_players].points[i])
                        #Se guarda cada carta en el array cartas
                        cartas.append(players[num_players].model.model.names[class_id])
                    except:
                        print("MISSED") #Si por un casual se detectan mas de 2 cartas, se desechan el resto
                        pass
                players_result.append((players[num_players].name, cartas, len(cartas))) #Se almacena en el array el nombre y las cartas de cada jugador

            #Si todos los jugadores (excluyendo el Crupier) tiene 2 cartas, se analiza las manos mas fuertes y se predice la probabilidad de ganar a futuro
            if all(num_cards == 2 for _, _, num_cards in players_result[1:]):
                pp_players = []
                try:
                    for player in players_result[1:]:
                        pp_players.append(pp.Player(player[0], pp.Card.of(*traduccion(player[1])))) #Se inicializa jugador y su mano correspondiente
                    manos = manoGanadora(pp_players, pp.Card.of(*traduccion(players_result[0][1]))) #Se obtiene los pesos de las manos
                    if simulacion_process == None or not simulacion_process.is_alive():
                        parent_conn, child_conn = Pipe() #Se define variables para comunicacion multiproceso
                        # Definicion multriproceso
                        simulacion_process = Process(target=probGanador, args=(child_conn, pp_players, pp.Card.of(*traduccion(players_result[0][1]))))
                        simulacion_process.start() #Se inicia multiproceso (Calcula la probabilidad de los jugadores a ganar a futuro)
                except:
                    print("No se ha podido calcular las manos o la probabilidad de ganar")
            
            try:
                if parent_conn.poll():
                    prob = parent_conn.recv()#Recupera la informacion de la cola del multiproceso
            except:
                print("No hay cola inicializada")

            #Se escribe todos los resultados en la imagen de la mesa
            for num_players in range(Jugador.NumeroJugadores):
                if len(manos) != 0:
                    datos1 = next(((categoria_mano, rango_mano) for nombre, categoria_mano, rango_mano in manos if nombre == players[num_players].name), ('', ''))
                else:
                    datos1 = {}
                if prob:
                    datos2 = next((datos for n, datos in prob if n == players[num_players].name), {})
                else:
                    datos2 = {}
                
                ganador_nombre = manos[0][0] if manos else None
                datos3 = "Ganador de la ronda actual" if ganador_nombre == players[num_players].name else ''

                if num_players != 0:
                    try:
                        text = f'{datos3}\n% Ganar a futuro: {datos2}\nCombinacion: {datos1[1]}\nMano ganadora: {datos1[0]}'
                    except:
                        text = ''
                    draw.text(players[num_players].textPoint, text, font=ImageFont.truetype("ariblk.ttf", 25), fill=(0, 0, 0))
            
            #Se muestra la imagen de la mesa con toda la informacion(Nombre de jugadores, cartas detectadas, manos ganadoras, probabilidades ....)
            cv2.imshow("Mesa", cv2.cvtColor(np.array(CopyPokerTable), cv2.COLOR_RGB2BGR))

            if (cv2.waitKey(30) == 27): #Si se presiona ESC, se sale de la ventana
                simulacion_process.kill()
                break 

#Init definition
if __name__ == "__main__":
    args = parse_arguments()
    main(parse_arguments())