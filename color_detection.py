import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
import os

# Load the dataset
def load_dataset(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File {file_path} tidak ditemukan.")
    df = pd.read_csv(file_path)
    X = df[['r', 'g', 'b']].values
    y = df['color_name'].values
    return X, y

# Preprocess the dataset for CNN
def preprocess_data(X, y):
    X_processed = X.reshape(-1, 1, 1, 3) / 255.0  # Normalize the RGB values
    color_dict = {color: idx for idx, color in enumerate(np.unique(y))}
    y_processed = np.array([color_dict[color] for color in y])
    y_processed = tf.keras.utils.to_categorical(y_processed, num_classes=len(color_dict))
    return X_processed, y_processed, color_dict

# Build the CNN model
def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (1, 1), activation='relu', input_shape=input_shape),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    global bgr_value, x_pos, y_pos, frame
    if event == cv2.EVENT_MOUSEMOVE:
        x_pos, y_pos = x, y
        bgr_value = frame[y, x]

# Main function
def main():
    global frame, bgr_value, x_pos, y_pos
    bgr_value = np.array([0, 0, 0])
    x_pos, y_pos = 0, 0

    # Path ke file colors.csv
    file_path = r'colors.csv'

    # Load dataset
    X, y = load_dataset(file_path)
    
    # Preprocess data
    X_processed, y_processed, color_dict = preprocess_data(X, y)
    
    # Build CNN model
    model = build_model((1, 1, 3), len(color_dict))
    
    # Train CNN model
    model.fit(X_processed, y_processed, epochs=100, verbose=0)
    
    # Reverse the color_dict for prediction
    reverse_color_dict = {idx: color for color, idx in color_dict.items()}

    # Open video capture
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Color Detection')
    cv2.setMouseCallback('Color Detection', mouse_callback)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame for faster processing
        resized_frame = cv2.resize(frame, (640, 480))

        # Update the frame for mouse callback
        frame = resized_frame

        # Convert BGR to RGB
        rgb_value = bgr_value[::-1] / 255.0  # BGR to RGB and normalize

        # Predict the color
        rgb_value = rgb_value.reshape(1, 1, 1, 3)
        prediction = model.predict(rgb_value)
        color_name = reverse_color_dict[np.argmax(prediction)]

        # Display the color name and RGB value on the frame
        cv2.putText(resized_frame, f'Color: {color_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(resized_frame, f'RGB: {tuple(bgr_value[::-1])}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Draw a circle at the cursor position
        cv2.circle(resized_frame, (x_pos, y_pos), 5, (255, 255, 255), -1)

        # Display the resulting frame
        cv2.imshow('Color Detection', resized_frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()