import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import datasets, layers, models


(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()#Veriler, verisetinden etiketlere yüklendi.
training_images, testing_images = training_images / 255, testing_images /255 #Veriler 0-255 arasında ölçeklendirilerek normalize edildi.

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck'] #Sınıflar belirlendi.
'''
for i in range(16):
   plt.subplot(4,4,i+1)
   plt.xticks([])
   plt.yticks([])                                            #Verisetinden 16 adet imgeyi gösteren kod
   plt.imshow(training_images[i], cmap=plt.cm.binary)
   plt.xlabel(class_names[training_labels[i][0]])

plt.show()
'''

training_images = training_images[:20000]    #20.000 örneği eğitim için aldık
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]       #4.000 örneği test için aldık
testing_labels = testing_labels[:4000]

'''
model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3))) #Giriş katmanı, matris filtre 3=RGB
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')

model.save('image_classifier.model')
'''
#'''
model = models.load_model('image_classifier.model')

img = cv.imread('f3_dog.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)

prediction = model.predict(np.array([img]) / 255)
index = np.argmax(prediction) # En çok uyarılan algılayıcıyı index numarasına atayan kod
print(f'Prediction is probably a {class_names[index]}') # Atanan index numarasını belirlediğimiz sınıfa atayan kod

plt.show()
#'''