import seaborn as sns

from Canvas import Canvas

sns.set()

img_height = 512
img_width = 512
channels = 3
CANVAS_SHAPE = (img_height, img_width, channels)

canvas = Canvas(CANVAS_SHAPE, "images/shape_1_1.jpg")
canvas.paste_to_canvas()