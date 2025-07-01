import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

# --- The Core Image Processing Function (Now with a strength parameter) ---
def correct_color_cast(image, strength=1.0):
    """
    Corrects a color cast in an image using a blend of the original
    and a Gray World corrected version.

    Args:
        image (np.ndarray): The input image in BGR format.
        strength (float): The amount of correction to apply (0.0 to 1.0).
                          1.0 is full correction, 0.0 is the original image.

    Returns:
        np.ndarray: The color-corrected image in BGR format.
    """
    # If strength is 0, no need to do any work.
    if strength == 0:
        return image
    
    # Perform the full Gray World correction first
    img_float = image.astype(np.float32)
    b, g, r = cv2.split(img_float)
    
    avg_b = np.mean(b)
    avg_g = np.mean(g)
    avg_r = np.mean(r)
    
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    
    if avg_b == 0 or avg_g == 0 or avg_r == 0:
        return image

    scale_b = avg_gray / avg_b
    scale_g = avg_gray / avg_g
    scale_r = avg_gray / avg_r
    
    corrected_b = b * scale_b
    corrected_g = g * scale_g
    corrected_r = r * scale_r
    
    full_corrected_img_float = cv2.merge([corrected_b, corrected_g, corrected_r])
    full_corrected_img = np.clip(full_corrected_img_float, 0, 255).astype(np.uint8)

    # If strength is 1.0, return the fully corrected image
    if strength == 1.0:
        return full_corrected_img

    # Otherwise, blend the original and the fully corrected image
    # final_image = (1 - strength) * original + strength * corrected
    blended_image = cv2.addWeighted(image, 1.0 - strength, full_corrected_img, strength, 0)
    
    return blended_image

# --- The GUI Application Class ---
class ColorCorrectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Free Color Cast Corrector")
        self.root.configure(bg='#2e2e2e')

        self.original_cv_image = None
        self.corrected_cv_image = None
        
        main_frame = tk.Frame(root, bg='#2e2e2e')
        main_frame.pack(padx=20, pady=20, fill="both", expand=True)

        image_frame = tk.Frame(main_frame, bg='#2e2e2e')
        image_frame.pack(pady=10, fill="both", expand=True)

        tk.Label(image_frame, text="Original Image", font=("Helvetica", 14, "bold"), fg="white", bg='#2e2e2e').grid(row=0, column=0, padx=10)
        tk.Label(image_frame, text="Corrected Image", font=("Helvetica", 14, "bold"), fg="white", bg='#2e2e2e').grid(row=0, column=1, padx=10)

        self.original_panel = tk.Label(image_frame, bg='#4a4a4a', text="Load an image to begin", fg="white", width=50, height=25)
        self.original_panel.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        self.corrected_panel = tk.Label(image_frame, bg='#4a4a4a', text="Correction will appear here", fg="white", width=50, height=25)
        self.corrected_panel.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        
        image_frame.grid_columnconfigure(0, weight=1)
        image_frame.grid_columnconfigure(1, weight=1)
        image_frame.grid_rowconfigure(1, weight=1)

        # --- NEW: Slider for correction strength ---
        control_frame = tk.Frame(main_frame, bg='#2e2e2e')
        control_frame.pack(fill='x', pady=5)
        
        tk.Label(control_frame, text="Correction Strength:", font=("Helvetica", 12), fg="white", bg='#2e2e2e').pack(side="left", padx=(10,5))
        
        self.strength_slider = tk.Scale(control_frame, from_=0, to=100, orient="horizontal", command=self.apply_correction, state="disabled", bg='#4a4a4a', fg='white', troughcolor='#3e3e3e', highlightthickness=0)
        self.strength_slider.set(90) # Default to 90% strength, which is less blue
        self.strength_slider.pack(fill='x', expand=True, side="left", padx=(5,10))

        button_frame = tk.Frame(main_frame, bg='#2e2e2e')
        button_frame.pack(pady=10)

        self.btn_load = tk.Button(button_frame, text="Load Image", command=self.load_image, font=("Helvetica", 12), bg="#4CAF50", fg="white", relief="flat", padx=10)
        self.btn_load.pack(side="left", padx=10)
        
        self.btn_save = tk.Button(button_frame, text="Save Corrected Image", command=self.save_image, state="disabled", font=("Helvetica", 12), bg="#f44336", fg="white", relief="flat", padx=10)
        self.btn_save.pack(side="left", padx=10)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        if not path: return

        try:
            self.original_cv_image = cv2.imread(path)
            if self.original_cv_image is None: raise ValueError("Could not read image file.")
            
            self.display_image(self.original_cv_image, self.original_panel)
            
            self.btn_save.config(state="normal", bg="#008CBA")
            self.strength_slider.config(state="normal") # Enable the slider
            self.apply_correction() # Apply the initial correction

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load or process image: {e}")

    def apply_correction(self, *args):
        if self.original_cv_image is None: return

        strength_val = self.strength_slider.get() / 100.0
        self.corrected_cv_image = correct_color_cast(self.original_cv_image, strength=strength_val)
        self.display_image(self.corrected_cv_image, self.corrected_panel)

    def display_image(self, cv_image, panel):
        panel_w, panel_h = 500, 500
        h, w, _ = cv_image.shape
        scale = min(panel_w/w, panel_h/h)
        nw, nh = int(w * scale), int(h * scale)
        
        resized_image = cv2.resize(cv_image, (nw, nh), interpolation=cv2.INTER_AREA)
        image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(image_rgb)
        img_tk = ImageTk.PhotoImage(image=img)
        
        panel.config(image=img_tk, text="")
        panel.image = img_tk

    def save_image(self):
        if self.corrected_cv_image is None: return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG file", "*.png"), ("JPEG file", "*.jpg")])
        if not path: return

        try:
            cv2.imwrite(path, self.corrected_cv_image)
            messagebox.showinfo("Success", f"Image saved successfully to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ColorCorrectorApp(root)
    root.mainloop()