#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np


# In[2]:


# Load the saved model
model = tf.keras.models.load_model('alz_model.h5')


# function to check if the file is an MRI image
def is_mri_image(file_path):
    # check the file extension to see if it is a valid image format
    valid_extensions = ['.jpg', '.jpeg', '.png']
    if not any(file_path.lower().endswith(ext) for ext in valid_extensions):
        return False
    
    # check if the image dimensions are valid for an MRI image
    image = Image.open(file_path)
    width, height = image.size
    if width != 176 or height != 208:
        return False
    
    # check if the image mode is valid for an MRI image
    if image.mode != 'L':
        return False
    
    # if all checks pass, then return True
    return True

# function to predict the class of an MRI image
def predict():
    class_names = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']
    global image_label, result_label
    file_path = filedialog.askopenfilename()
    image = tf.keras.preprocessing.image.load_img(file_path)
    image = tf.image.resize(image, (128, 128))
    input_array = tf.keras.preprocessing.image.img_to_array(image)
    input_array = np.array([input_array])
    pred = model.predict(input_array)
    res = np.argmax(pred)
    re1 = class_names[res]
    result_label.configure(text="Predicted class: " + re1, font=("Helvetica", 16))
    
    # Display the image
    img = Image.open(file_path)
    img = img.resize((150, 150))
    photo = ImageTk.PhotoImage(img)
    image_label.configure(image=photo)
    image_label.image = photo

root = tk.Tk()
root.title("Image Classification")
root.geometry("1920x1080")

title_label = tk.Label(root, text="Alzheimer's Disease Classification", font=("Helvetica", 32))
title_label.pack(pady=10)

select_button = tk.Button(root, text="Select Image", font=("Helvetica", 18), command=predict)
select_button.pack()

image = Image.open(r"C:\Users\Dhanush A\Downloads\Project Code\Brain-Scan-Concept-Animation.gif")
frames = []
try:
    while True:
        frames.append(ImageTk.PhotoImage(image))
        image.seek(len(frames))
except EOFError:
    pass

image_label = tk.Label(root)
image_label.pack(pady=10)

result_label = tk.Label(root, text="")
result_label.pack(pady=10)

root.mainloop()


# In[ ]:




