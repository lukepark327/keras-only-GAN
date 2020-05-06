import os
import math
import numpy as np
import matplotlib.pyplot as plt


def plot_images(generator,
                noise_input,
                show=False,
                step=0,
                name="gan"):

    os.makedirs(name, exist_ok=True)
    filename = os.path.join(name, "%05d.png" % step)

    images = generator.predict(noise_input)

    plt.figure(figsize=(2.2, 2.2))
    num_images = images.shape[0]
    image_size = images.shape[1]
    rows = int(math.sqrt(noise_input.shape[0]))

    for i in range(num_images):
        plt.subplot(rows, rows, i + 1)
        image = np.reshape(images[i], [image_size, image_size])
        plt.imshow(image, cmap='gray')
        plt.axis('off')

    plt.savefig(filename)

    if show:
        plt.show()
    else:
        plt.close('all')
