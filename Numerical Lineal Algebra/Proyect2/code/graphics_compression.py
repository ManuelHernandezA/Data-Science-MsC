from datetime import datetime
from matplotlib.image import imsave
from imageio import imread
import numpy as np

def image_compression(img_path, start=1, stop=41, step=5):
    # datetime object containing current date and time
    now = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    image = imread(img_path)
    im_new = np.zeros(image.shape)
    for c in range(start, stop, step):   
        tmp = []
        for i in range(3):
            U, s, V = np.linalg.svd(image[:,:,i].astype(float)/256)
            im_new[:,:,i] = np.matrix(U[:,:c])*np.diag(s[:c])*np.matrix(V[:c,:])
            frob = np.linalg.norm(im_new[:,:,i],'fro')/np.linalg.norm(image[:,:,i].astype(float)/256,'fro')*100
            tmp.append(frob)
        name = "image_"+now+f"{frob:.02f}".replace(".","-" )+".jpeg"
        imsave(name, np.clip(im_new,0,1))
        print("\tsaved at "+ name)

## Test the compression
if __name__ == '__main__':
    print("Compressing image...")
    image_compression("lenna.png")