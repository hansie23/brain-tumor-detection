from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate


def unet_model(input_shape):
    inputs = Input(input_shape)
    
    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        
    # Mid-level
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    drop3 = Dropout(0.5)(conv3)
        
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(drop3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D((2, 2))(drop4)
    
    # Decoder
    upconv1 = UpSampling2D(size=(2, 2))(pool4)
    upconv1 = Conv2D(128, 2, activation='relu', padding='same')(upconv1)
    merge1 = concatenate([conv3, upconv1], axis=3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(merge1)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(conv4)
        
    upconv2 = UpSampling2D(size=(2, 2))(conv4)
    upconv2 = Conv2D(64, 2, activation='relu', padding='same')(upconv2)
    merge2 = concatenate([conv2, upconv2], axis=3)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(merge2)
        
    upconv3 = UpSampling2D(size=(2, 2))(conv5)
    upconv3 = Conv2D(64, 2, activation='relu', padding='same')(upconv3)
    merge3 = concatenate([conv1, upconv3], axis=3)
    conv6 = Conv2D(64, 3, activation='relu', padding='same')(merge3)
        
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(conv6)
    
    outputs = Conv2D(1, 1, activation='sigmoid')(conv7)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model
