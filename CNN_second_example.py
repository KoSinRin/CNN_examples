# Импортируем библиотеки 
import tensorflow as tf

# Создаем модель CNN
model = tf.keras.models.Sequential()

# Добавляем слой свертки с 32 картами признаков и окном свертки 3x3
model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))

# Добавляем слой пулинга с окном 2x2
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Добавляем слой свертки с 64 картами признаков и окном свертки 3x3
model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))

# Добавляем слой пулинга с окном 2x2
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Преобразуем данные из 2D в 1D формат
model.add(tf.keras.layers.Flatten())

# Добавляем полносвязный слой с 128 нейронами
model.add(tf.keras.layers.Dense(128, activation='relu'))

# Добавляем выходной слой с количеством классов, равным количеству классов в наборе данных
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

# Компилируем модель
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучаем модель
model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_val, y_val))

# Оцениваем точность модели на тестовых данных
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



