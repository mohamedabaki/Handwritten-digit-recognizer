import tkinter as tk
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from tensorflow import keras

# Load the trained model
model = keras.models.load_model('model.h5')

# Create a Tkinter GUI window
window = tk.Tk()
window.title("Handwritten Digit Recognition")

# Create a canvas for drawing digits
canvas_width = 280
canvas_height = 280
canvas = tk.Canvas(window, width=canvas_width, height=canvas_height, bg='white')
canvas.pack()

# Create a label to display the predicted digit
label = tk.Label(window, text="Draw a digit", font=("Arial", 20))
label.pack()

# Function to predict the digit based on the drawing
def predict_digit():
    # Get the image from the canvas
    image = Image.new('L', (canvas_width, canvas_height), 'white')
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, canvas_width, canvas_height], fill='white')
    canvas.postscript(file='tmp.eps', colormode='color')
    img = Image.open('tmp.eps').convert('L')
    img = img.resize((28, 28))

    # Preprocess the image
    #img = img.convert('L')
    #This line inverts the grayscale image, reversing the intensity values of the pixels.
    img = ImageOps.invert(img)
    image_array = np.array(img)
    image_array = image_array.astype('float32') / 255.0
    #It prepares the image for prediction by reshaping it to have dimensions (1, 28, 28, 1)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = np.expand_dims(image_array, axis=-1)

    # Predict the digit using the trained model
    prediction = model.predict(image_array)
    predicted_digit = np.argmax(prediction)

    # Display the predicted digit
    messagebox.showinfo("Prediction", f"The predicted digit is: {predicted_digit}")

# Function to clear the canvas
def clear_canvas():
    canvas.delete('all')
    label.config(text="Draw a digit")

# Function to handle mouse dragging on the canvas
def draw(event):
    x, y = event.x, event.y
    r = 5  # Radius of the drawing point
    canvas.create_oval(x - r, y - r, x + r, y + r, fill='black')

# Bind the drawing function to the left mouse button motion
canvas.bind("<B1-Motion>", draw)

# # Create a button to predict the digit
predict_button = tk.Button(window, text="Predict", font=("Arial", 14), command=predict_digit)
predict_button.pack(side=tk.LEFT, padx=10, pady=10)

# Create a button to clear the canvas
clear_button = tk.Button(window, text="Clear", font=("Arial", 14), command=clear_canvas)
clear_button.pack(side=tk.RIGHT, padx=10, pady=10)

# Run the GUI main loop
window.mainloop()
