import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np

data = np.load('face_images.npz')['face_images']
data = np.swapaxes(np.swapaxes(data, 1, 2), 0, 1)
train_images = data[49:,:].reshape(7000, 96, 96, 1)/255

'''
(train, label), (test, label_test) = tf.keras.datasets.mnist.load_data()

train_images = train.reshape(train.shape[0], 28, 28, 1)
train_images = (train_images-127.5)/127.5

array = np.zeros((6265, 28, 28, 1))
k = 0
for i in range(len(label)):
  if label[i] == 7:
    array[k,:] = train_images[i,:]
    k = k+1
train_images = array[65:,:]
print(train_images.shape)
'''


BUFFER_SIZE = train_images.shape[0]
BATCH_SIZE = 100
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).batch(BATCH_SIZE)

plt.imshow(train_images[0])
plt.show()
def discriminator():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(7, (3,3), padding='same', input_shape=(96, 96, 1)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.LeakyReLU(0.1))
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    return model

dis_model = discriminator()
disc_opt = tf.optimizers.Adam()

def discriminator_loss(y_pred_real, y_pred_fake):
    real = tf.sigmoid(y_pred_real)
    fake = tf.sigmoid(y_pred_fake)
    real_loss = tf.keras.losses.BinaryCrossentropy()(tf.ones_like(real), real)
    fake_loss = tf.keras.losses.BinaryCrossentropy()(tf.zeros_like(fake), fake)
    return fake_loss+real_loss

def generator():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(24*24*256, input_shape=(100,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Reshape((24, 24, 256)))
    model.add(tf.keras.layers.Conv2DTranspose(128, (3, 3), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2DTranspose(64, (3,3), strides=(2,2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2DTranspose(1, (3,3), strides=(2,2), padding='same'))
    return model

gen_model = generator()
gen_opt = tf.optimizers.Adam()

def generator_loss(fake_pred):
    loss = tf.sigmoid(fake_pred)
    fake_loss = tf.keras.losses.BinaryCrossentropy()(tf.ones_like(loss), loss)
    return fake_loss

def train(dataset, epoch):
    for j in range(epoch):
        for images in dataset:
            images = tf.cast(images, tf.dtypes.float32)
            train_steps(images)

def train_steps(images):
    fake_noise = np.random.randn(BATCH_SIZE, 100).astype('float32')
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = gen_model(fake_noise)

        fake_output = dis_model(generated_images)
        real_output = dis_model(images)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

        gen_gradient = gen_tape.gradient(gen_loss, gen_model.trainable_variables)
        disc_gradient = disc_tape.gradient(disc_loss, dis_model.trainable_variables)

        disc_opt.apply_gradients(zip(disc_gradient, dis_model.trainable_variables))
        gen_opt.apply_gradients(zip(gen_gradient, gen_model.trainable_variables))

        print('disc_loss: ', np.mean(disc_loss))
        print('gen_loss: ', np.mean(gen_loss))


train(train_dataset, 2)


plt.imshow(tf.reshape(gen_model(np.random.randn(1, 100)), (96, 96)))
plt.show()
