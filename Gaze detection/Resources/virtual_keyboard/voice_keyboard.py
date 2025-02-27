import pyttsx3
import keyboard

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Function to speak a character
def speak_character(char):
    engine.say(char)
    engine.runAndWait()

# Main loop to detect key presses
print("Start typing (press 'esc' to exit):")
