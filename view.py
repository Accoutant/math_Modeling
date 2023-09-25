from d2l import torch as d2l
import numpy as np
import os
import matplotlib.pyplot as plt

root_path = ("./NJU_CPOL_update2308/dBZ/1.0km/data_dir_235"
             )
path_list = os.listdir(root_path)

images = []
for path in path_list:
    image = np.load(root_path+ '/' + str(path))
    images.append(image)

d2l.show_images(images, 5, 5, scale=5)
plt.show()