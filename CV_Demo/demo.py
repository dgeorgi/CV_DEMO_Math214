import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import cv2
import sys

# Default image path
default_image_path = '/dist/puppies.jpg'

# Use the first argument if provided, else use the default image path
image_path = sys.argv[1] if len(sys.argv) > 1 else default_image_path

original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
original_image = cv2.resize(original_image, (300, 300))  # Resize for convenience

# Initialize the main window
root = tk.Tk()
root.title("Convolution Demo")
root.geometry("1200x600")  # Adjust the window size

large_font = ('Verdana', 12)  # Define a larger font
kernel_display = tk.Label(root, text="Select a Kernel", width=20, height=10, font=large_font)

# Display original image
fig_original = Figure(figsize=(6, 6), dpi=100)
canvas_original = FigureCanvasTkAgg(fig_original, master=root)
plot_original = fig_original.add_subplot(111)
plot_original.imshow(original_image, cmap='gray')
canvas_original.get_tk_widget().grid(row=0, column=0)

# Kernel matrix display (placeholder for now)
kernels = {
    "Identity": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
    "Edge Detection": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
    "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    "Gaussian Blur": (1/16) * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]),
    "Blur": (1/9) * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
    "Sobel X": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    "Sobel Y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
}

kernel_display = tk.Label(root, text="Select a Kernel", width=20, height=10)
kernel_display.grid(row=0, column=1)

# Display resulting image (placeholder for now)
fig_result = Figure(figsize=(6, 6), dpi=100)
canvas_result = FigureCanvasTkAgg(fig_result, master=root)
plot_result = fig_result.add_subplot(111)
plot_result.imshow(np.zeros_like(original_image), cmap='gray')  # Empty image for now
canvas_result.get_tk_widget().grid(row=0, column=2)

# Dropdown for kernel selection
kernel_options = ['Kernel 1', 'Kernel 2', 'Kernel 3']  # Replace with actual kernels
kernel_var = tk.StringVar()
kernel_dropdown = ttk.Combobox(root, textvariable=kernel_var, values=kernel_options)
kernel_dropdown.grid(row=1, column=1)
kernel_dropdown.current(0)

def apply_convolution(image, kernel):
    return cv2.filter2D(image, -1, kernel)

# Update kernel options in the GUI
kernel_options = list(kernels.keys())
kernel_var = tk.StringVar()
kernel_dropdown = ttk.Combobox(root, textvariable=kernel_var, values=kernel_options)
kernel_dropdown.grid(row=1, column=1)
kernel_dropdown.current(0)

# Function to update the convoluted image and kernel matrix display
def update_image(*args):
    selected_kernel = kernels[kernel_var.get()]
    convoluted_image = apply_convolution(original_image, selected_kernel)
    plot_result.imshow(convoluted_image, cmap='gray')
    canvas_result.draw()

    # Update the kernel matrix display
    kernel_matrix_text = '\n'.join([' '.join(map(str, row)) for row in selected_kernel])
    kernel_display.config(text=kernel_matrix_text)

# Bind the update function to the dropdown
kernel_var.trace("w", update_image)

# Variables to track the position of the box
box_x, box_y = 0, 0  # Starting position

def draw_box():
    # Clear previous drawings
    plot_original.clear()
    plot_original.imshow(original_image, cmap='gray')

    # Draw a new box at (box_x, box_y)
    kernel_size = kernels[kernel_var.get()].shape[0]  # Assuming square kernel
    rect = plt.Rectangle((box_x, box_y), kernel_size, kernel_size, edgecolor='red', facecolor='none')
    plot_original.add_patch(rect)
    
    canvas_original.draw()

# Bind arrow key events
root.bind("<Left>", lambda e: move_box(-1, 0))
root.bind("<Right>", lambda e: move_box(1, 0))
root.bind("<Up>", lambda e: move_box(0, -1))
root.bind("<Down>", lambda e: move_box(0, 1))

def calculate_dot_product():
    global box_x, box_y

    kernel = kernels[kernel_var.get()]
    kernel_size = kernel.shape[0]

    # Ensure the box does not go outside the image
    max_x = original_image.shape[1] - kernel_size
    max_y = original_image.shape[0] - kernel_size
    box_x = min(max(box_x, 0), max_x)
    box_y = min(max(box_y, 0), max_y)

    # Extract the image region
    image_region = original_image[box_y:box_y + kernel_size, box_x:box_x + kernel_size]

    # Create a formula representation
    formula_representation = ""
    for i in range(kernel_size):
        for j in range(kernel_size):
            formula_part = f"({kernel[i, j]}*{image_region[i, j]})"
            if j < kernel_size - 1:
                formula_part += " + "
            formula_representation += formula_part
        formula_representation += "\n"

    # Calculate dot product
    dot_product = np.sum(kernel * image_region)
    
    # Update the kernel matrix display with the dot product calculation
    calculation_text = f"Formula:\n{formula_representation}\nResult: {dot_product}"
    kernel_display.config(text=calculation_text)

def draw_green_box():
    # Clear previous drawings on the result plot
    plot_result.clear()
    selected_kernel = kernels[kernel_var.get()]
    convoluted_image = apply_convolution(original_image, selected_kernel)
    plot_result.imshow(convoluted_image, cmap='gray')

    # Draw a new green box at (box_x, box_y)
    kernel_size = kernels[kernel_var.get()].shape[0]  # Assuming square kernel
    rect = plt.Rectangle((box_x, box_y), kernel_size, kernel_size, edgecolor='green', facecolor='none')
    plot_result.add_patch(rect)
    
    canvas_result.draw()

def move_box(delta_x, delta_y):
    global box_x, box_y
    box_x += delta_x
    box_y += delta_y
    draw_box()       # Draw red box on original image
    draw_green_box() # Draw green box on result image
    calculate_dot_product()

# Initial draw of the green box on the resulting image
draw_green_box()


root.focus_set()  # Make sure the root window has focus for key events

update_image()

root.mainloop()
