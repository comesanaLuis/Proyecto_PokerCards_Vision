from PIL import Image, ImageTk
import cv2
import numpy as np

def combinar_imagenes():
    # Cargar las imágenes
    imagen1 = Image.open('PokerTable.jpg')
    imagen2 = Image.open('Cartas/1.jpeg')

    # Crear una imagen combinada
    imagen1.paste(imagen2)

    # Convertir la imagen combinada a formato numpy
    imagen_np = np.array(imagen1)

    # Mostrar la imagen combinada en una ventana de OpenCV
    cv2.imshow("Imagen Combinada", cv2.cvtColor(imagen_np, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

combinar_imagenes()
