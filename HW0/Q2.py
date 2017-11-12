from PIL import Image
import numpy as np

lena = Image.open("lena.png")
lena_modified = Image.open("lena_modified.png")

w, h = lena.size
for j in range(h):
    for i in range(w):
        if lena.getpixel((i, j)) == lena_modified.getpixel((i, j)):
            lena_modified.putpixel((i, j), 255)

lena_modified.show()
lena_modified.save("ans_two.png")


    

# print(w, h, a.shape, b.shape)

# lena.show()
# lena_modified.show()

