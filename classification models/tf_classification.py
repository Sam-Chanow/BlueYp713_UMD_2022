import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

#loading dataset

df=pd.read_csv('matrix_dataset.csv', sep=',', low_memory=False)
df = df.drop(columns=['index']) #get rid of useless numbering from matrix i.e. 0th column
print(df.head())
df.info(verbose=True)
df = df.dropna()
df.info(verbose=True)

tf.keras.backend.clear_session()


print(max(df['event_sub_type']))
#Split the data for labels and training/testing

X = df.drop(columns="event_sub_type")
Y = df['event_sub_type']

#setup better y labels using a 17 dim array
ln = len(Y)
print(ln)
new_Y = np.zeros((ln,17))
for i, label in enumerate(Y):

    new_Y[i][label] = 1
    #new_Y.append(l)

#print(new_Y)


X_train, X_test, y_train, y_test = train_test_split(
    X, new_Y,
    test_size=0.2, random_state=42
)

print(y_train)

#exit(0)

#Scale the datasets

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Now the actual model
tf.random.set_seed(42)

print(X_test_scaled)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu'),
    #tf.keras.layers.Dense(256, activation='relu'),
    #tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(17, activation='sigmoid')
])

model.compile(
    loss='mse',
    optimizer=tf.keras.optimizers.Adam(lr=0.03),
    metrics=[
        #tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        #tf.keras.metrics.Loss(name='loss'),
        tf.keras.metrics.CategoricalCrossentropy(name='loss'),
        tf.keras.metrics.Accuracy(name='accuracy'),
        #tf.keras.metrics.Precision(name='precision'),
        #tf.keras.metrics.Recall(name='recall')
    ]
)

#Saving a figure

history = model.fit(X_train_scaled, y_train, epochs=100, validation_data=(X_test_scaled, y_test))

from matplotlib import rcParams

rcParams['figure.figsize'] = (18, 8)
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False

plt.plot(
    np.arange(1, 101),
    history.history['loss'], label='Loss'
)

plt.plot(
    np.arange(1, 101),
    history.history['val_loss'], label='validation Loss'
)
plt.plot(
   np.arange(1, 31),
    history.history['accuracy'], label='Accuracy'
)
#plt.plot(
#    np.arange(1, 31),
#    history.history['precision'], label='Precision'
#)
#plt.plot(
#    np.arange(1, 31),
#    history.history['recall'], label='Recall'
#)
plt.title('Evaluation metrics', size=20)
plt.xlabel('Epoch', size=14)
plt.legend();
plt.savefig("tf_test.png")
