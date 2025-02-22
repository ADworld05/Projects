import tkinter as tk
import pygame
import os

# Initialize pygame for sound
pygame.mixer.init()

def play_sound(char):
    """Plays sound for the selected key."""
    try:
        sound_file = f"sounds/{char}.mp3"
        if os.path.exists(sound_file):  # Ensure the file exists before playing
            pygame.mixer.music.load(sound_file)
            pygame.mixer.music.play()
        else:
            print(f"Sound file not found: {sound_file}")
    except Exception as e:
        print(f"Error playing sound for {char}: {e}")

def insert_char(char):
    """Inserts the clicked character into the input field."""
    input_field.insert(tk.END, char)

def backspace():
    """Deletes the last character in the input field."""
    input_field.delete(len(input_field.get()) - 1, tk.END)

# Create the main application window
root = tk.Tk()
root.title("Custom Virtual Keyboard")
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

# Create buttons and store references
key_buttons = []
for row, key_row in enumerate(keys, start=1):
    button_row = []
    for col, key in enumerate(key_row):
        btn = tk.Button(
            root, text=key, font=("Arial", 14), width=4, height=2, bg="lightgray",
            command=lambda char=key: insert_char(char))
        btn.grid(row=row, column=col, padx=5, pady=5)
        button_row.append(btn)

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
    global current_row, current_col
    key_buttons[current_row][current_col].config(bg="lightgray")
    current_row, current_col = new_row, new_col
    key_buttons[current_row][current_col].config(bg="yellow")

    # Get the character at the new position and play the sound
    char = key_buttons[current_row][current_col].cget("text")
    if char == "__":
        char = "space"
    elif char == "⌫":
        char = "backspace"
    
    play_sound(char)

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