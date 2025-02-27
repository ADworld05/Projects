import tkinter as tk
from tkinter import messagebox

def submit():
    selected_sensitivity = sensitivity_var.get()
    if selected_sensitivity:
        messagebox.showinfo("Selection", f"You selected: {selected_sensitivity}")
        root.destroy()
    else:
        messagebox.showwarning("Warning", "Please select a sensitivity level")

# Create main window
root = tk.Tk()
root.title("Sensitivity Selection Wizard")
root.geometry("300x200")

tk.Label(root, text="Select sensitivity level:", font=("Arial", 12)).pack(pady=10)

sensitivity_var = tk.StringVar(value="")

tk.Radiobutton(root, text="High", variable=sensitivity_var, value="High").pack(anchor="w", padx=20)
tk.Radiobutton(root, text="Medium", variable=sensitivity_var, value="Medium").pack(anchor="w", padx=20)
tk.Radiobutton(root, text="Low", variable=sensitivity_var, value="Low").pack(anchor="w", padx=20)

tk.Button(root, text="Submit", command=submit).pack(pady=20)

root.mainloop()
