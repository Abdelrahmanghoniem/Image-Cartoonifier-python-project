import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import easygui
import tkinter as tk
from tkinter import filedialog, Label, Button, TOP
from PIL import ImageTk, Image
import sys

# Function to upload image using easygui
def upload():
    # Open a file dialog to select an image
    ImagePath = easygui.fileopenbox()
    # Call the cartoonify function with the selected image path
    cartoonify(ImagePath)

# Function to cartoonify the image
def cartoonify(ImagePath):
    # Read the image
    img = cv.imread(ImagePath)

    # Exit if image is not selected
    if img is None:
        print("Can not find any image. Choose appropriate file")
        sys.exit()

    # Resize the image
    resize = cv.resize(img, (480, 360), interpolation=cv.INTER_AREA)
    # Convert the resized image to grayscale
    gray = cv.cvtColor(resize, cv.COLOR_BGR2GRAY)
    # Apply Gaussian blur to the grayscale image
    blurred = cv.GaussianBlur(gray, (9, 3), 0)
    # Detect edges using adaptive thresholding
    edge = cv.adaptiveThreshold(
        blurred, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 7, 7)
    # Apply median blur to the resized image
    reblur = cv.medianBlur(resize, 5)

    # Function to quantize the image
    def quan(img, k):
        data = np.float32(img).reshape(-1, 5)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 15, 0.01)
        ret, label, center = cv.kmeans(
            data, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        result = center[label.flatten()]
        result = result.reshape(img.shape)
        return result

    # Call the quantization function
    q = quan(reblur, 70)
    # Apply bilateral filter to the quantized image
    noise = cv.bilateralFilter(q, 15, 190, 190)
    # Mask the filtered image with the detected edges
    ci = cv.bitwise_and(noise, noise, mask=edge)

    # Plotting
    plt.figure(figsize=(6, 4), dpi=250)
    plt.subplot(1, 2, 1)
    plt.imshow(cv.cvtColor(resize, cv.COLOR_BGR2RGB))
    plt.title('Original image')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 2, 2)
    plt.imshow(cv.cvtColor(ci, cv.COLOR_BGR2RGB))
    plt.title('Cartoonified')
    plt.xticks([])
    plt.yticks([])
    plt.show()

    plt.figure(figsize=(5, 4), dpi=210)
    plt.subplot(2, 4, 1)
    plt.imshow(cv.cvtColor(resize, cv.COLOR_BGR2RGB))
    plt.title("Original")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 4, 2)
    plt.imshow(cv.cvtColor(gray, cv.COLOR_BGR2RGB))
    plt.title('Gray image')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 4, 3)
    plt.imshow(cv.cvtColor(blurred, cv.COLOR_BGR2RGB))
    plt.title('Blurred gray')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 4, 4)
    plt.imshow(cv.cvtColor(edge, cv.COLOR_BGR2RGB))
    plt.title('Detected edges')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 4, 5)
    plt.imshow(cv.cvtColor(reblur, cv.COLOR_BGR2RGB))
    plt.title('Blurred BGR')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 4, 6)
    plt.imshow(cv.cvtColor(q, cv.COLOR_BGR2RGB))
    plt.title('Quantised')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 4, 7)
    plt.imshow(cv.cvtColor(noise, cv.COLOR_BGR2RGB))
    plt.title('Filtered')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 4, 8)
    plt.imshow(cv.cvtColor(ci, cv.COLOR_BGR2RGB))
    plt.title('Masked')
    plt.xticks([])
    plt.yticks([])
    plt.show()

# Tkinter GUI setup
top = tk.Tk()
top.geometry('600x400')
top.title('Image Cartoonifier')
top.configure(background='white')
label = Label(top, background='#15d6b9', font=('calibri', 18, 'bold'))

# Button to upload image
upload = Button(top, text="Upload Image", command=upload, padx=10, pady=5)
upload.configure(background='#2dc8e3', foreground='white', font=('Cooper Std Black', 18, 'bold'))
upload.pack(side=TOP, pady=160)

# Run the GUI
top.mainloop()
