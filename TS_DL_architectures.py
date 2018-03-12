from keras import layers, models, optimizers

class TS_CNN():
    """

    """

    def __init__(self):
        print ('building CNN...')

    def create_model(self, X_shape, nr_classes, learn_rate, dropout=0, last_layer='gl_avg_pooling'):

        # Common CNN part
        X = layers.Input(X_shape)
        conv1 = layers.Conv1D(filters=128, kernel_size=7, strides=1, padding='same', activation='relu')(X)
        conv1 = layers.normalization.BatchNormalization()(conv1)
        conv1 = layers.MaxPooling1D(pool_size=2, strides=2)(conv1)
        conv1 = layers.Dropout(dropout)(conv1)

        conv2 = layers.Conv1D(filters=256, kernel_size=5, strides=1, padding='same', activation='relu')(conv1)
        conv2 = layers.normalization.BatchNormalization()(conv2)
        conv2 = layers.MaxPooling1D(pool_size=2, strides=2)(conv2)
        conv2 = layers.Dropout(dropout)(conv2)

        conv3 = layers.Conv1D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(conv2)
        conv3 = layers.normalization.BatchNormalization()(conv3)
        conv3 = layers.MaxPooling1D(pool_size=2, strides=2)(conv3)
        conv3 = layers.Dropout(dropout)(conv3)

        # Global average pooling
        if last_layer == 'gl_avg_pooling':
            full = layers.pooling.GlobalAveragePooling1D()(conv3)

        # Fully Connected Layer
        elif last_layer == 'fully_connected':
            full = layers.Flatten()(conv3)
            full = layers.Dense(units=128, activation='relu')(full)
            full = layers.Dropout(dropout)(full)

        # Long Short Term Memory layer
        elif last_layer == 'LSTM':
            lstm1 = layers.LSTM(units=128)(conv3)
            full = layers.Dropout(dropout)(lstm1)

        Y_prob = layers.Dense(units=nr_classes, activation='softmax')(full)

        model = models.Model(inputs=X, outputs=Y_prob)

        # Define optimizer and compile
        adam = optimizers.Adam(lr=learn_rate)
        model.compile(optimizer=adam,
                      loss='categorical_crossentropy',
                      metrics=['acc'])

        return model