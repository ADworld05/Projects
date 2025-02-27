import tkinter as tk
from tkinter import messagebox

def submit():
    try:
        selected_sensitivity = float(sensitivity_var.get())
        messagebox.showinfo("Selection", f"You selected sensitivity: {selected_sensitivity}")
        root.destroy()
    except ValueError:
        messagebox.showwarning("Warning", "Please enter a valid numerical value")

# Create main window
root = tk.Tk()
root.title("Sensitivity Input Wizard")
root.geometry("300x200")

tk.Label(root, text="Enter sensitivity level (numeric):", font=("Arial", 12)).pack(pady=10)

sensitivity_var = tk.StringVar()

entry = tk.Entry(root, textvariable=sensitivity_var)
entry.pack(pady=5)

tk.Button(root, text="Submit", command=submit).pack(pady=20)

root.mainloop()
