from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

def build_model(num_classes):
    """
    Builds a disaster classification model using ResNet50 as a backbone.

    Args:
        num_classes (int): The number of disaster categories.

    Returns:
        A compiled Keras model.
    """
    # Load the ResNet50 model with pre-trained ImageNet weights, excluding the top classification layer
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

    # Fine-tune from this layer onwards
    fine_tune_at = 143  # Unfreeze the top 2 conv blocks

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.3)(x) # Reduced dropout
    x = Dense(512, activation='relu')(x) # Added another dense layer
    predictions = Dense(num_classes, activation='softmax')(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

if __name__ == '__main__':
    # Example of how to use the build_model function
    num_classes = 5  # Example: 5 disaster types
    model = build_model(num_classes)
    model.summary()