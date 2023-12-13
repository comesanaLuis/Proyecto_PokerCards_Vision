#Librerias
import cv2
import argparse
from ultralytics import YOLO
from ultralytics import data
import supervision as sv

#Definicion de argumentos
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO_PokerCards_Vision")
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--predict", action='store_true')
    parser.add_argument("--valid", action='store_true')
    parser.add_argument("--webcam-resoution",
        default = [1280, 720],
        nargs = 2,
        type = int
    )
    args = parser.parse_args()
    return args

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
        #Definicion de VideoInput
        cap = cv2.VideoCapture(4)
        frame_width, frame_height = args.webcam_resoution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
 
        #Modelo YOLOv8 de predición entrenado
        model = YOLO("YOLO_PokerCards_Vision_Final.pt")

        #Definicion de propiedades Enmarcador de objetos detectados
        box_annotator = sv.BoxAnnotator(
            thickness=2,
            text_thickness=2,
            text_scale=1
        )
        
        while True:
            ret, frame = cap.read() #Se extrae frame
            result = model(frame, agnostic_nms = False)[0] #Se predice objetos del frame

            detections = sv.Detections.from_ultralytics(result) #Se extraen a variables atributos de cada objeto detectado
            print(detections.class_id)
            numero = list(set(detections.class_id))
            print(str(numero))

            #Definicion de cada una de las etiquetas a mostrar en box_annotator
            labels = [
                f"{model.model.names[class_id]} {conficence:0.2f}"
                for conficence, class_id in zip(detections.confidence, detections.class_id) # %de confianza y nombre
            ]

            #Se imprimen en el Video output los objetos detectados mediante su encuadre y etiqueta
            frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

            cv2.imshow("YOLO_PokerCards_Vision0", frame) #Se muestra VideoOutput
    
            if (cv2.waitKey(30) == 27):#Si se presiona ESC, se sale de la ventana
                break

#Init definition
if __name__ == "__main__":
    args = parse_arguments()
    main(parse_arguments())