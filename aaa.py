from PIL import ImageFont

# Obtener una lista de las fuentes predeterminadas
font_list = ImageFont._Font.mro()
print(font_list)