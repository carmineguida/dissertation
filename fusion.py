import keras
from keras import regularizers
from keras.models import load_model, Model
from keras.layers import BatchNormalization, Concatenate, Dense, Dropout, Input


def load_pretrained_model(path, which):
    model = load_model(path)
    model.trainable = False
    for layer in model.layers:
        layer._name = which + '_' + layer._name
    return model


def find_layer(model, name):
    for layer in model.layers:
        if (layer.name == name):
            return layer
    return None

# Stage 1: Load models
xray = load_pretrained_model('saved/xray.h5', 'xray')
mri = load_pretrained_model('saved/mri.h5', 'mri')

xray_first_layer = xray.layers[0]
mri_first_layer = mri.layers[0]

# Stage 2: Get intermediate layers, add layers and combine.
xray_intermediate = find_layer(xray, 'xray_avg_pool')
mri_intermediate = find_layer(mri, 'mri_global_max_pooling3d_1')

X_xray = xray_intermediate.output
X_xray = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(X_xray)
X_xray = Dropout(0.40)(X_xray)

X_mri = mri_intermediate.output
X_mri = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(X_xray)
X_mri = Dropout(0.40)(X_xray)

combined = Concatenate()([X_mri, X_xray])

# Stage 3: Add Clinical, additional layers and final softmax layer
clinical_input = Input(shape=(4, ))
X_clinical = clinical_input
clinical = Model(inputs=clinical_input, outputs=X_clinical, name='clinical')

combined = Concatenate()([combined, X_clinical])

X = combined
X = BatchNormalization()(X)
for i in range(3):
    X = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01))(X)
    X = Dropout(0.40)(X)

X = Dense(5, activation='softmax')(X)

inputs = [mri_first_layer.input, xray_first_layer.input, clinical.layers[0].input]
outputs = [X]

fusion_model = Model(inputs=inputs, outputs=outputs)

# Save fusion model summary
with open('fusion_summary.txt', 'w') as f:
    fusion_model.summary(print_fn=lambda x:f.write(x + '\n'))

