#highlighting the keys with arrow keys

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
root.geometry("720x440")

# Input field
input_field = tk.Entry(root, font=("Arial", 18), width=40)
input_field.grid(row=0, column=0, columnspan=10, pady=10)

# Keyboard layout
keys = [
    ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],  # Row 1
    ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p"],  # Row 2
    ["a", "s", "d", "f", "g", "h", "j", "k", "l"],       # Row 3
    ["z", "x", "c", "v", "b", "n", "m"],                # Row 4
]

# Dictionary to store button references for highlighting
key_buttons = []
for row, key_row in enumerate(keys):
    button_row = []
    for col, key in enumerate(key_row):
        btn = tk.Button(
            root,
            text=key,
            font=("Arial", 14),
            width=4,
            height=2,
            bg="lightgray",
            command=lambda char=key: insert_char(char),
        )
        btn.grid(row=row + 1, column=col, padx=5, pady=5)
        button_row.append(btn)
    key_buttons.append(button_row)

# Add Spacebar and Backspace buttons separately
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

# Add space and backspace to the button grid for navigation
key_buttons.append([space_button, backspace_button])

# Initial highlighted key
current_row, current_col = 0, 0
key_buttons[current_row][current_col].config(bg="yellow")

def move_highlight(new_row, new_col):
    """Move the highlight to the new position."""
    global current_row, current_col
    key_buttons[current_row][current_col].config(bg="lightgray")  # Reset previous
    current_row, current_col = new_row, new_col
    key_buttons[current_row][current_col].config(bg="yellow")  # Highlight new

def handle_keypress(event):
    """Handle arrow key presses to move the highlighted key."""
    global current_row, current_col
    if event.keysym == "Left" and current_col > 0:
        move_highlight(current_row, current_col - 1)
    elif event.keysym == "Right" and current_col < len(key_buttons[current_row]) - 1:
        move_highlight(current_row, current_col + 1)
    elif event.keysym == "Up" and current_row > 0:
        move_highlight(current_row - 1, min(current_col, len(key_buttons[current_row - 1]) - 1))
    elif event.keysym == "Down" and current_row < len(key_buttons) - 1:
        move_highlight(current_row + 1, min(current_col, len(key_buttons[current_row + 1]) - 1))

# Bind arrow keys to movement
root.bind("<Left>", handle_keypress)
root.bind("<Right>", handle_keypress)
root.bind("<Up>", handle_keypress)
root.bind("<Down>", handle_keypress)

# Run the application
root.mainloop()
