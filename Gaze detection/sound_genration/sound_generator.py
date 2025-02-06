from gtts import gTTS
import os

# Characters for the keyboard
keys = [
    ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
    ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
    ["k", "l", "m", "n", "o", "p", "q", "r", "s"],
    ["t", "u", "v", "w", "x", "y", "z", "space", "backspace"]
]

# Create "sounds" directory if it doesn't exist
os.makedirs("sounds", exist_ok=True)

# Generate audio files
for row in keys:
    for char in row:
        text = char if char not in ["space", "backspace"] else ("space" if char == "space" else "backspace key")
        tts = gTTS(text=text, lang="en")
        tts.save(f"sounds/{char}.mp3")
        print(f"Generated: {char}.mp3")
