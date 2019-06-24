from skimage import io
from sklearn.cluster import KMeans
import numpy as np

# Tiago Moreira Trocoli da Cunha
# Number: 226078
 
def reduce_colors(name, numberOfCluters):
    
    image = io.imread(name + ".png")

    # reshape the image to have rows = rows*cols, columns = 3. 
    rows = image.shape[0]
    cols = image.shape[1]
    image = image.reshape(rows*cols,3)
    
    # apply k-means
    kmeans = KMeans(n_clusters = numberOfCluters, n_init=10, max_iter=200)
    kmeans.fit(image)
    
    # get the coordinates of clusters and turn into numpy array
    clusters = np.asarray(kmeans.cluster_centers_,dtype=np.uint8)
    # get the label for each pixel
    labels = np.asarray(kmeans.labels_,dtype=np.uint8 )
    # turn labels' shape equal to image's  
    labels = labels.reshape(rows,cols)
    
    np.save("codebook_"+name+str(numberOfCluters)+".npy", clusters)
    io.imsave("compressed_"+name+str(numberOfCluters)+".png", labels)
    
    # create a RGB image
    image = np.zeros((labels.shape[0],labels.shape[1],3),dtype=np.uint8 )
    
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            image[i,j,:] = clusters[labels[i,j],:]

    io.imsave(name + str(numberOfCluters) + ".png",image);
    
    print(name + " reduced to " + str(numberOfCluters) + " colors!")    
    
def main():
    
    names   = ["baboon", "monalisa", "peppers", "watch"]
    numbers = [16, 32, 64, 128] 
    
    for name in names:
        for number in numbers:
            reduce_colors(name, number)
    
if __name__ == "__main__":
    main()