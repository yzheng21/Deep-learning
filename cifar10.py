import inception
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from IPython.display import Image,display
#inception.maybe_download()

model=inception.Inception()

def classify(image_path):
    display(Image(image_path))
    pred=model.classify(image_path=image_path)
    model.print_scores(pred=pred,k=10,only_first_name=True)

image_path=os.path.join(inception.data_dir,'cropped_panda.jpg')
classify(image_path)
print("------------------------------")

classify(image_path='images/parrot.jpg')
def plot_realized_image(image_path):
    resized_image=model.get_resized_image(image_path=image_path)
    plt.imshow(resized_image,interpolation='nearest')
    plt.show()

plot_realized_image(image_path='images/parrot.jpg')
print("-------------------------------")
classify(image_path='images/parrot_cropped3.jpg')
print("-------------------------------")
classify(image_path='images/elon_musk.jpg')
plot_realized_image(image_path='images/elon_musk.jpg')
model.close()