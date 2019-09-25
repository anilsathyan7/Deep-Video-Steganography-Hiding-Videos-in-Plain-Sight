import sys
from keras.models import Model
from keras.layers import Dense, Input, concatenate, Conv2D
from keras.models import load_model
from keras.utils import plot_model

# Load complete model
model=load_model(sys.argv[1], compile=False)

# Print model summary
model.summary()

# Generate hiding network
encoder=Model([model.get_layer('secret').input,model.get_layer('cover').input],model.get_layer('hide_conv_f').output)
encoder.save("hide.h5")

# Save model plot
plot_model(encoder, to_file='hide.png')

# Create new input layer
new_ip=Input(shape=(224,224,3))

# Reveal network [Re-initialize layer and weights]
rconv_3x3=Conv2D(50, kernel_size=3, padding="same", activation='relu', name='revl_conv3x3_1')(new_ip)
rconv_3x3=Conv2D(50, kernel_size=3, padding="same", activation='relu', name='revl_conv3x3_2')(rconv_3x3)
rconv_3x3=Conv2D(50, kernel_size=3, padding="same", activation='relu', name='revl_conv3x3_3')(rconv_3x3)
rconv_3x3=Conv2D(50, kernel_size=3, padding="same", activation='relu', name='revl_conv3x3_4')(rconv_3x3)

rconv_4x4=Conv2D(50, kernel_size=4, padding="same", activation='relu', name='revl_conv4x4_1')(new_ip)
rconv_4x4=Conv2D(50, kernel_size=4, padding="same", activation='relu', name='revl_conv4x4_2')(rconv_4x4)
rconv_4x4=Conv2D(50, kernel_size=4, padding="same", activation='relu', name='revl_conv4x4_3')(rconv_4x4)
rconv_4x4=Conv2D(50, kernel_size=4, padding="same", activation='relu', name='revl_conv4x4_4')(rconv_4x4)

rconv_5x5=Conv2D(50, kernel_size=5, padding="same", activation='relu', name='revl_conv5x5_1')(new_ip)
rconv_5x5=Conv2D(50, kernel_size=5, padding="same", activation='relu', name='revl_conv5x5_2')(rconv_5x5)
rconv_5x5=Conv2D(50, kernel_size=5, padding="same", activation='relu', name='revl_conv5x5_3')(rconv_5x5)
rconv_5x5=Conv2D(50, kernel_size=5, padding="same", activation='relu', name='revl_conv5x5_4')(rconv_5x5)

rconcat_1 = concatenate([rconv_3x3,rconv_4x4,rconv_5x5], axis=3, name="revl_concat_1")

rconv_5x5=Conv2D(50, kernel_size=5, padding="same", activation='relu', name='revl_conv5x5_f')(rconcat_1)
rconv_4x4=Conv2D(50, kernel_size=4, padding="same", activation='relu', name='revl_conv4x4_f')(rconcat_1)
rconv_3x3=Conv2D(50, kernel_size=3, padding="same", activation='relu', name='revl_conv3x3_f')(rconcat_1)

rconcat_f1 = concatenate([rconv_5x5,rconv_4x4,rconv_3x3], axis=3, name="revl_concat_2")

secret_pred = Conv2D(3, kernel_size=1, padding="same", name='revl_conv_f')(rconcat_f1)

# Generate reveal network
decoder=Model(new_ip,secret_pred)

# Load weights from the parent model 
decoder.load_weights(sys.argv[1], by_name=True)
decoder.save('reveal.h5')

#Save model plot
plot_model(decoder, to_file='reveal.png')

'''
Sample run:-
python split_model.py checkpoints/steg_model-06-0.03.hdf5
'''
