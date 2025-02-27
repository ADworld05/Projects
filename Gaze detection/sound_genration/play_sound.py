import pygame

# Initialize pygame for sound
pygame.mixer.init()

def play_sound(char):
    """Plays sound for the selected key."""
    try:
        sound_file = f"sounds/{char}.mp3"
        if os.path.exists(sound_file):
            pygame.mixer.music.load(sound_file)
            pygame.mixer.music.play()
    except Exception as e:
        print(f"Error playing sound for {char}: {e}")

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
