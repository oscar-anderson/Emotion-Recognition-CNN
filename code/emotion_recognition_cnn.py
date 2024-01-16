# Emotion-Recognition Convolutional Neural Network.

# Import dependencies.
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import History
import matplotlib.pyplot as plt

# Define functions.
def load_data(train_data_path: str, test_data_path: str) -> tuple:
    """
    Load, preprocess and augment data.

    Parameters:
    - train_data_path: Path to training data folder.
    - test_data_path: Path to testing data folder.

    Returns:
    Tuple containing training and testing data generators.
    """
    train_data_generator_gen = ImageDataGenerator(
        rescale = 1. / 255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        height_shift_range = 0.2
    )

    test_data_gen = ImageDataGenerator(rescale = 1. / 255)

    train_data_generator = train_data_generator_gen.flow_from_directory(
        train_data_path,
        target_size = (48, 48),
        batch_size = 32,
        color_mode = 'grayscale',
        class_mode = 'categorical'
    )

    test_data_generator = test_data_gen.flow_from_directory(
        test_data_path,
        target_size = (48, 48),
        batch_size = 32,
        color_mode = 'grayscale',
        class_mode = 'categorical'
    )

    return train_data_generator, test_data_generator

def create_CNN() -> Sequential:
    """
    Create convolutional neural network.

    Returns:
    Compiled Keras Sequential model.
    """
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (48, 48, 1)))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    model.add(Conv2D(128, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    model.add(Flatten())

    model.add(Dense(192, activation = 'relu'))

    model.add(Dense(7, activation = 'softmax'))

    model.compile(
        loss ='categorical_crossentropy',
        optimizer = 'adam',
        metrics = ['accuracy']
        )

    return model

def train_model(model: Sequential, train_data_generator, test_data_generator, epochs: int) -> History:
    """
    Train the neural network.

    Parameters:
    - model: Compiled Keras Sequential model.
    - train_data_generator: Data to be used for training.
    - test_data_generator: Data to be used for validation.
    - epochs: Number of epochs for training.

    Returns:
    Keras History variable containing training metrics.
    """
    
    history = model.fit(
        train_data_generator,
        epochs = epochs,
        validation_data = test_data_generator
        )
    
    return history

def evaluate_model(model: Sequential, test_data_generator) -> tuple:
    """
    Evaluate the performance of the trained model on the test data.

    Parameters:
    - model: Compiled Keras Sequential model.
    - test_data_generator: Data to be used for validation.

    Returns:
    Tuple containing loss and accuracy.
    """
    evaluation = model.evaluate(test_data_generator)
    loss = evaluation[0]
    accuracy = evaluation[1]
    
    return loss, accuracy

def plot_performance(history: History) -> None:
    """
    Plot the training and validation metrics.

    Parameters:
    - history: Keras History variable containing training metrics.
    """
    plt.figure(figsize = (12, 6))
    plt.suptitle('Emotion-Recognition Convolutional Neural Network Performance')

    plt.subplot(1, 2, 1)
    plt.plot(history.epoch, history.history['accuracy'], label = 'Training', marker = 'o')
    plt.plot(history.epoch, history.history['val_accuracy'], label = 'Validation', marker = 'o')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.epoch, history.history['loss'], label = 'Training', marker = 'o')
    plt.plot(history.epoch, history.history['val_loss'], label = 'Validation', marker = 'o')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Call functions to run model.
train_data_path = 'C:/Users/Oscar/Documents/Projects/fer2013_emotion_detection/train'
test_data_path = 'C:/Users/Oscar/Documents/Projects/fer2013_emotion_detection/test'

train_data_generator, test_data_generator = load_data(train_data_path, test_data_path)
model = create_CNN()
history = train_model(model, train_data_generator, test_data_generator, epochs = 40)
loss, accuracy = evaluate_model(model, test_data_generator)

print(f'Loss: {loss:.4f}%')
print(f'Test Accuracy: {accuracy * 100:.2f}%')

plot_performance(history)
