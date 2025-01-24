#highlighting the keys periodically

import tkinter as tk

def insert_char(char):
    """Inserts the clicked character into the input field."""
    current_text = input_field.get()
    input_field.delete(0, tk.END)
    input_field.insert(0, current_text + char)

def backspace():
    """Deletes the last character in the input field."""
    current_text = input_field.get()
    input_field.delete(0, tk.END)
    input_field.insert(0, current_text[:-1])

# Create the main application window
root = tk.Tk()
root.title("QWERTY Virtual Keyboard")
root.geometry("650x450")

# Input field
input_field = tk.Entry(root, font=("Arial", 18), width=40)
input_field.grid(row=0, column=0, columnspan=10, pady=10)

# Keyboard layout
keys = [
    # Row 1
    ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
    # Row 2
    ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p"],
    # Row 3
    ["a", "s", "d", "f", "g", "h", "j", "k", "l"],
    # Row 4
    ["z", "x", "c", "v", "b", "n", "m"],
]

# Dictionary to store button references for highlighting
key_buttons = []

# Add keys to the keyboard
for row, key_row in enumerate(keys):
    button_row = []
    for col, key in enumerate(key_row):
        btn = tk.Button(
            root,
            text=key,
            font=("Arial", 14),
            width=4,
            height=2,
            bg="lightgray",  # Default color
            command=lambda char=key: insert_char(char),
        )
        btn.grid(row=row + 1, column=col, padx=2, pady=8)
        button_row.append(btn)
    key_buttons.extend(button_row)

# Add Spacebar and Backspace buttons
space_button = tk.Button(
    root,
    text="Space",
    font=("Arial", 14),
    width=15,
    height=2,
    bg="lightgray",
    command=lambda: insert_char(" "),
)
space_button.grid(row=5, column=0, columnspan=5, pady=5)
key_buttons.append(space_button)

backspace_button = tk.Button(
    root,
    text="Backspace",
    font=("Arial", 14),
    width=15,
    height=2,
    bg="lightgray",
    command=backspace,
)
backspace_button.grid(row=5, column=5, columnspan=5, pady=5)
key_buttons.append(backspace_button)

# Function to highlight keys periodically
def highlight_keys(index=0):
    if index > 0:
        # Reset the previous button's color
        key_buttons[index - 1].config(bg="lightgray")

    if index < len(key_buttons):
        # Highlight the current button
        key_buttons[index].config(bg="yellow")
        # Schedule the next highlight after 1.5 seconds (1500 milliseconds)
        root.after(1500, highlight_keys, index + 1)
    else:
        # Restart the highlighting cycle
        key_buttons[-1].config(bg="lightgray")
        root.after(1500, highlight_keys, 0)

# Start highlighting the keys
highlight_keys()

# Run the application
root.mainloop()

