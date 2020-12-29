import numpy as np
import cv2, math
import mnist #Get data set from
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from scipy import ndimage
from keras.models import Sequential # Artificial Neural Network Architecture
from keras.layers import Dense # Layers in ANN
from keras.utils import to_categorical

#Load Data set
train_images = mnist.train_images() # training data images
train_labels = mnist.train_labels() # training data labels
test_images = mnist.test_images() # testing data images
test_labels = mnist.test_labels() # testing data labels


# normalize the images. Normalize the pixel values from [0, 255] to [-0.5, 0.5] 
# to make network easier to train
train_images = (train_images/255) - 0.5
test_images = (test_images/255) - 0.5
# flatten the images. Flatten each 28x28 image into a 28^2 = 784 dimentional vector to 
# pass into the neural network
train_images = train_images.reshape((-1,784))
test_images = test_images.reshape((-1,784))
print(train_images.shape) # (60000, 784)
print(test_images.shape) # (10000, 784)


# Build the Model
# 3 layers, 2 layers with 64 neurons and the relu function
# 1 layer with 10 neurons and softmax function

model = Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add( Dense(64, activation = 'relu', input_dim = 784))
model.add( Dense(64, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))


# Compile the model
# The loss function measures how well the model did on training and then tries 
# to improve on it using the optimizer

model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy', #(Classes that are greater than 2)
    metrics = ['accuracy']
)


# Train the model
model.fit(
    train_images,
      to_categorical(train_labels), # Ex. 2 it expects [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
      epochs = 5, #The number of iterations over the entire dataset to train on
      batch_size = 32 # the number of samles per gradient update for training
)

# model.save('my_model.h5')


#Evaluate the model
model.evaluate(
    test_images,
      to_categorical(test_labels)  
)


# df = pd.read_csv('/content/drive/My Drive/Numbers NN/Book1.csv')
# df4 = {"Real_Number": 4}
# df = df.append(df4, ignore_index=True)
# print(df)
# x = tf.zeros([0, 784])

# model = tf.keras.models.load_model('model.h5')

imgC = cv2.imread('Python\Big100.jpg')
imgGS = cv2.imread('Python\Big100.jpg', cv2.IMREAD_GRAYSCALE)
# shpe = np.array(imgGS.shape)
# imgGS = cv2.resize(imgGS, )
thresh = 180
imgBW = cv2.threshold(255-imgGS, thresh, 255, cv2.THRESH_BINARY)[1]
digit_image = -np.ones(imgGS.shape)
print(digit_image)
plt.imshow(imgBW, cmap = "gray")
plt.show()

def getBestShift(imgCD):
    cy,cx = ndimage.measurements.center_of_mass(imgCD)
    print(cy, cx)

    rows,cols = imgCD.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty

def shift(imgCD,sx,sy):
    rows,cols = imgCD.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(imgCD,M,(cols,rows))
    return shifted

    while np.sum(imgCD[0]) == 0:
        imgCD = imgCD[1:]

    while np.sum(imgCD[:,0]) == 0:
        imgCD = np.delete(imgCD,0,1)

    while np.sum(imgCD[-1]) == 0:
        imgCD = imgCD[:-1]

    while np.sum(imgCD[:,-1]) == 0:
        imgCD = np.delete(imgCD,-1,1)

    rows,cols = imgCD.shape

    if rows > cols:
        factor = 20.0/rows
        rows = 20
        cols = int(round(cols*factor))
        imgCD = cv2.resize(imgCD, (cols,rows))
    else:
        factor = 20.0/cols
        cols = 20
        rows = int(round(rows*factor))
        imgCD = cv2.resize(imgCD, (cols, rows))
        
    colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
    rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
    imgCD = np.lib.pad(imgCD,(rowsPadding,colsPadding),'constant')


height, width = imgBW.shape
tl = []
br = []
c = False
# print(imgBW.shape)
for cropped_width in range(100, 500, 20):
    for cropped_height in range(100, 500, 20):
        for shift_x in range(0, int(width-cropped_width), int(cropped_width/4)):
            for shift_y in range(0, int(height-cropped_height), int(cropped_height/4)):
              imgCD = imgBW[shift_y:shift_y+cropped_height,shift_x:shift_x + cropped_width]
              if np.count_nonzero(imgCD) <= 20:
                continue
              if (np.sum(imgCD[0]) != 0) or (np.sum(imgCD[:,0]) != 0) or (np.sum(imgCD[-1]) != 0) or (np.sum(imgCD[:,-1]) != 0):
                continue
              top_left = np.array([shift_y, shift_x])
              bottom_right = np.array([shift_y+cropped_height, shift_x + cropped_width])
              # if len(tl) > 0:
                # print(tl[-2], top_left[0], tl[-1], top_left[1])
              for i in range(len(tl)-1):
                # print(tl, "tl")
                # print(top_left, "Top Left")
                if (tl[i] == top_left[0]) and (tl[i+1] == top_left[1]) and (br[i] == bottom_right[0]) and (br[i+1] == bottom_right[1]):
                  c = True
                  print("ooga booga")
              if c == True:
                c = False
                continue
              tl = np.append(tl, top_left)
              br = np.append(br, bottom_right)
              
              while np.sum(imgCD[0]) == 0:
                  top_left[0] += 1
                  imgCD = imgCD[1:]

              while np.sum(imgCD[:,0]) == 0:
                  top_left[1] += 1
                  imgCD = np.delete(imgCD,0,1)

              while np.sum(imgCD[-1]) == 0:
                  bottom_right[0] -= 1
                  imgCD = imgCD[:-1]

              while np.sum(imgCD[:,-1]) == 0:
                  bottom_right[1] -= 1
                  imgCD = np.delete(imgCD,-1,1)
                
              actual_w_h = bottom_right-top_left
              if (np.count_nonzero(digit_image[top_left[0]:bottom_right[0],top_left[1]:bottom_right[1]]+1) >
                          0.2*actual_w_h[0]*actual_w_h[1]):
                  continue

              rows,cols = imgCD.shape
              compl_dif = abs(rows-cols)
              half_Sm = compl_dif/2
              half_Big = half_Sm if half_Sm*2 == compl_dif else half_Sm+1
              half_Sm = int(round(half_Sm))
              half_Big = int(round(half_Big))
              # print(half_Sm, half_Big)
              if rows > cols:
                  imgCD = np.lib.pad(imgCD,((0,0),(half_Sm,half_Big)),'constant')
              else:
                  imgCD = np.lib.pad(imgCD,((half_Sm,half_Big),(0,0)),'constant')

              imgCD = cv2.resize(imgCD, (20, 20))
              imgCD = np.lib.pad(imgCD,((4,4),(4,4)),'constant')


              shiftx,shifty = getBestShift(imgCD)
              shifted = shift(imgCD,shiftx,shifty)
              imgCD = shifted

              flatten = imgCD.flatten() / 255.0
              # print(flatten.shape)
              imgCD = imgCD.reshape((1,784))
              
              # prediction = [tf.reduce_max(y),tf.argmax(y,1)[0]]
              prediction = model.predict(imgCD)
              
              pred = np.argmax(prediction)
              print(np.argmax(prediction))
              # pred = sess.run(prediction, feed_dict={x: [flatten]})
              # pred = feed_dict = {x: [flatten]}
              print(prediction)
              # pred = prediction
              # print(tl, br)
              # digit_image[top_left[0]:bottom_right[0],top_left[1]:bottom_right[1]] = pred[0]
              cv2.rectangle(imgC,tuple(top_left[::-1]),tuple(bottom_right[::-1]),color=(0,255,0),thickness=5)

              font = cv2.FONT_HERSHEY_SIMPLEX
              # digit we predicted
              cv2.putText(imgC,f'{pred}',(top_left[1],bottom_right[0]+100),
                          font,fontScale=3,color=(0,255,0),thickness=4)
              # percentage
              # cv2.putText(imgC,format(pred[0]*100,".1f")+"%",(top_left[1]+30,bottom_right[0]+60),
              #             font,fontScale=0.8,color=(0,255,0),thickness=2)
plt.imshow(imgCD)
plt.imshow(imgC)
plt.show()


