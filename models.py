import tensorflow as tf
from tensorflow.keras import layers

def get_top_stream():
    input_shape = (32, 128, 1)
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), input_shape=input_shape, padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(1024))
    return (model)


def get_bottom_stream():
    
    model = None
    model = tf.keras.Sequential()
    input_shape = (32, None, 1)
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), input_shape=input_shape, padding='same'))
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3),  padding='same'))
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3),  padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3),  padding='same'))
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3),  padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(layers.Conv2D(filters=512, kernel_size=(3, 3),  padding='same'))
    model.add(layers.Conv2D(filters=512, kernel_size=(3, 3),  padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    padding = [[0, 0], [0, 0], [2, 2], [0, 0]]   
    model.add(layers.Conv2D(filters=1024, kernel_size=(4, 4), padding=padding))
    
    return (model)

def get_middle_stream(Ns):
    model = None
    model = tf.keras.Sequential()
    model.add(layers.Dense(Ns, input_shape = (None, None, 1024), activation=tf.nn.relu))
    return (model)

def get_character_error_rate(word1, word2):
    rows = len(word1) + 1
    cols = len(word2) + 1
    error_matrix = [[0 for i in range(cols)] for j in range(rows)]

    for i in range(len(error_matrix[0])):
        error_matrix[0][i] = i

    for i in range(len(error_matrix[0])):
        error_matrix[i][0] = i

    for i in range(1, rows):
        for j in range(1, cols):
            a = error_matrix[i-1][j] + 1
            b = error_matrix[i][j-1] + 1
            if word1[i-1] == word2[j-1]:
                c = error_matrix[i-1][j-1]
            else:
                c = error_matrix[i-1][j-1] + 2
            error_matrix[i][j] = min(a, b, c)
            
    return (error_matrix[rows-1][cols-1])

def norm_loss(out, target):
    return (tf.norm(target - out))


def get_optimizer():
    optimizer = tf.keras.optimizers.RMSprop()
    return (optimizer)
