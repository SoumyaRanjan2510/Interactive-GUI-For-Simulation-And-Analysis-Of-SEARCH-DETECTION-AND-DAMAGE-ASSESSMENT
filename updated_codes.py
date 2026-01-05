import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.special import i0
import math
import os
from math import comb


# ------------------ Main Application ------------------
class CombinedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Military Operations Research")
        self.root.configure(bg='#f0f0f5') 


        # ---------- Style ---------- 
        style = ttk.Style() 
        style.theme_use('clam') 
        style.configure('TNotebook.Tab', font=('Segoe UI', 11, 'bold'), padding=[10, 5]) 
        style.configure('TLabel', font=('Segoe UI', 11), background='#f0f0f5', foreground='#003366') 
        style.configure('TButton', font=('Segoe UI', 10), background='#4CAF50', foreground='white') 
        style.configure('TFrame', background='#f0f0f5')


        #----------- Tab Control ---------- 
        self.tab_control = ttk.Notebook(self.root) 
        self.tab_control.pack(expand=1, fill='both', padx=10, pady=10)

        window_width = 1400
        window_height = 680
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        center_x = int(screen_width / 2 - window_width / 2)
        center_y = int(screen_height / 2 - window_height / 2)
        root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        root.geometry("1200x750+425+200") 
        self.root.configure(bg='#f0f0f5') 

        self.tab_control = ttk.Notebook(root)

        self.detection_tab = ttk.Frame(self.tab_control)
        self.search_model_tab = ttk.Frame(self.tab_control)
        self.hit_probability_tab = ttk.Frame(self.tab_control)
        self.damage_tab= ttk.Frame(self.tab_control)
        self.salvo_tab= ttk.Frame(self.tab_control)
        self.single_vs_multiple_tab = ttk.Frame(self.tab_control)
        self.shooting_tactics_tab = ttk.Frame(self.tab_control)
        

        self.tab_control.add(self.detection_tab, text="Detection Theory")
        self.tab_control.add(self.search_model_tab, text="Search Model Detection")
        self.tab_control.add(self.hit_probability_tab, text="Hit Probability")
        self.tab_control.add(self.damage_tab, text="Damage Assessment")
        self.tab_control.add(self.salvo_tab, text="Salvo & Pattern Firing")
        self.tab_control.add(self.single_vs_multiple_tab, text="Single vs Multiple Aiming Points")
        self.tab_control.add(self.shooting_tactics_tab, text="Shooting Tactics")
        self.tab_control.pack(expand=1, fill='both')

        self.create_detection_theory_gui(self.detection_tab)
        self.create_search_model_gui(self.search_model_tab)
        self.create_hit_probability_gui(self.hit_probability_tab)
        self.create_damage_assessment_gui(self.damage_tab)
        self.create_salvo_pattern_gui(self.salvo_tab)
        self.create_single_vs_multiple_gui(self.single_vs_multiple_tab)
        self.create_shooting_tactics_gui(self.shooting_tactics_tab)

        

    # ---------------- Detection Theory GUI ----------------
    def create_detection_theory_gui(self, tab):
        class DetectionTheoryGUI:
            def __init__(self, root):
                self.root = root
                self.create_widgets()

            def create_widgets(self):
                input_frame = ttk.LabelFrame(self.root, text="Input Parameters")
                input_frame.pack(side=tk.LEFT, fill=tk.Y, expand=False, padx=10, pady=10)

                ttk.Label(input_frame, text="Probability of detection per scan (d):").grid(row=0, column=0, sticky=tk.W)
                self.d_entry = ttk.Entry(input_frame)
                self.d_entry.grid(row=0, column=1)
                self.d_entry.insert(0, "0.2")  # Default value

                ttk.Label(input_frame, text="Number of scans (n):").grid(row=1, column=0, sticky=tk.W)
                self.n_entry = ttk.Entry(input_frame)
                self.n_entry.grid(row=1, column=1)
                self.n_entry.insert(0, "5")  # Default value

                ttk.Label(input_frame, text="Mean time to detection (T):").grid(row=2, column=0, sticky=tk.W)
                self.t_entry = ttk.Entry(input_frame)
                self.t_entry.grid(row=2, column=1)
                self.t_entry.insert(0, "10")  # Default value

                ttk.Label(input_frame, text="Time duration for continuous scan (t):").grid(row=3, column=0, sticky=tk.W)
                self.small_t_entry = ttk.Entry(input_frame)
                self.small_t_entry.grid(row=3, column=1)
                self.small_t_entry.insert(0, "6")  # Default value

                ttk.Button(input_frame, text="Calculate", command=self.calculate).grid(row=4, column=0, columnspan=2, pady=10)

                self.result_label = ttk.Label(input_frame, text="", foreground="blue")
                self.result_label.grid(row=5, column=0, columnspan=2, pady=5)

                try:
                    base_dir = os.path.dirname(__file__)
                    image_path = os.path.join(base_dir, "detection.png")
                    image = Image.open(image_path)
                    image = image.resize((480, 430), Image.Resampling.LANCZOS)
                    self.help_img = ImageTk.PhotoImage(image)
                    help_label = tk.Label(input_frame, image=self.help_img)
                    help_label.grid(row=6, column=0, columnspan=2, pady=10)
                except Exception as e:
                    error_label = tk.Label(input_frame, text=f"Image could not be loaded:\n{e}", fg="red", justify=tk.LEFT)
                    error_label.grid(row=6, column=0, columnspan=2, pady=10, sticky=tk.W)

                plot_frame = ttk.LabelFrame(self.root, text="Detection Probability Plot")
                plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

                self.figure = plt.Figure(figsize=(7, 5), dpi=100)
                self.ax = self.figure.add_subplot(111)
                self.canvas = FigureCanvasTkAgg(self.figure, plot_frame)
                self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            def calculate(self):
                try:
                    d = float(self.d_entry.get())
                    n = int(self.n_entry.get())
                    T = float(self.t_entry.get())
                    small_t = float(self.small_t_entry.get())

                    if not (0 <= d <= 1):
                        raise ValueError("Probability d must be between 0 and 1.")
                    if n <= 0:
                        raise ValueError("Number of scans n must be a positive integer.")
                    if T <= 0:
                        raise ValueError("Mean time to detection T must be greater than 0.")
                    if small_t < 0:
                        raise ValueError("Time duration t cannot be negative.")

                    pn = 1 - (1 - d) ** n
                    pt = 1 - math.exp(-small_t / T)

                    self.result_label.config(
                        text=f"Pn (scanning) = {pn:.4f} | P(t={small_t}) (continuous) = {pt:.4f}"
                    )

                    time_values = np.linspace(0.1, max(10 * T, n), 300)
                    pt_curve_values = [1 - math.exp(-x / T) for x in time_values]
                    pn_curve_values = [1 - (1 - d) ** x for x in time_values]

                    self.ax.clear()
                    self.ax.plot(time_values, pt_curve_values, label="Continuous: P(t)", color='blue', linestyle='--', linewidth=2)
                    self.ax.plot(time_values, pn_curve_values, label="Scanning: Pn", color='green', linestyle='-', linewidth=2)
                    self.ax.set_title("Detection Probability vs. Time / Number of Scans", fontsize=12)
                    self.ax.set_xlabel("Time / Number of Scans", fontsize=10)
                    self.ax.set_ylabel("Probability of Detection", fontsize=10)
                    self.ax.set_ylim(0, 1.05)
                    self.ax.grid(True, linestyle=':', alpha=0.7)
                    self.ax.legend(loc="lower right", fontsize=9)
                    self.canvas.draw()

                except ValueError as ve:
                    messagebox.showerror("Input Error", str(ve))
                except Exception as e:
                    messagebox.showerror("Error", f"An unexpected error occurred:\n{str(e)}")

        DetectionTheoryGUI(tab)

    # ---------------- Search Model Detection GUI ----------------
    def create_search_model_gui(self, tab):
        def calculate_probability():
            try:
                A = float(area_entry.get())
                W = float(width_entry.get())
                x = float(length_entry.get())

                if A <= 0:
                    raise ValueError("Area (A) must be greater than zero.")
                if W < 0 or x < 0:
                    raise ValueError("Sweep Width (W) and Track Length (x) must be non-negative.")

                C = (W * x) / A
                random_prob = 1 - np.exp(-C)
                exhaustive_prob = min(C, 1)
                inverse_prob = C / (1 + C)
                ratio = exhaustive_prob / inverse_prob if inverse_prob > 0 else float('inf')

                #show results
                coverage_label.config(text=f"Coverage Factor (C): {C:.2f}")
                random_label.config(text=f"Random Search: {random_prob:.3f}")
                exhaustive_label.config(text=f"Exhaustive Search: {exhaustive_prob:.3f}")
                inverse_label.config(text=f"Inverse Law: {inverse_prob:.3f}")
                ratio_label.config(text=f"Overestimation Ratio (Exhaustive / Inverse): {ratio:.2f}")

                #plot
                C_values = np.linspace(0, C * 1.5 if C > 0 else 1, 500)
                random_values = 1 - np.exp(-C_values)
                exhaustive_values = np.clip(C_values, 0, 1)
                inverse_values = C_values / (1 + C_values)

                ax.clear()
                ax.plot(C_values, random_values, label='Random Search (1 - e^{-C})', color='blue', linestyle='--')
                ax.plot(C_values, exhaustive_values, label='Exhaustive Search (min(C, 1))', color='green')
                ax.plot(C_values, inverse_values, label='Inverse Law (C / (1 + C))', color='orange')
                ax.set_title("Probability of Detection vs. Coverage Factor")
                ax.set_xlabel("Coverage Factor (C)")
                ax.set_ylabel("Detection Probability")
                ax.set_ylim([0, 1.05])
                ax.legend()
                ax.grid(True)
                canvas.draw()

            except ValueError as ve:
                messagebox.showerror("Invalid Input", f"Error: {ve}")
                coverage_label.config(text="Coverage Factor (C): --")
                random_label.config(text="Random Search: --")
                exhaustive_label.config(text="Exhaustive Search: --")
                inverse_label.config(text="Inverse Law: --")
                ratio_label.config(text="Overestimation Ratio(Exh/Inv): --")

        frame = ttk.Frame(tab)
        frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.Y)

        ttk.Label(frame, text="Area (A)").grid(row=0, column=0, sticky="w")
        area_entry = ttk.Entry(frame)
        area_entry.grid(row=0, column=1)
        area_entry.insert(0, "200")

        ttk.Label(frame, text="Sweep Width (W)").grid(row=1, column=0, sticky="w")
        width_entry = ttk.Entry(frame)
        width_entry.grid(row=1, column=1)
        width_entry.insert(0, "20")

        ttk.Label(frame, text="Track Length (x)").grid(row=2, column=0, sticky="w")
        length_entry = ttk.Entry(frame)
        length_entry.grid(row=2, column=1)
        length_entry.insert(0, "30")

        calculate_button = ttk.Button(frame, text="Calculate", command=calculate_probability)
        calculate_button.grid(row=3, column=0, columnspan=2, pady=10)

        coverage_label = ttk.Label(frame, text="Coverage Factor (C): --")
        coverage_label.grid(row=4, column=0, columnspan=2, pady=5)

        random_label = ttk.Label(frame, text="Random Search: --")
        random_label.grid(row=5, column=0, columnspan=2)

        exhaustive_label = ttk.Label(frame, text="Exhaustive Search: --")
        exhaustive_label.grid(row=6, column=0, columnspan=2)

        inverse_label = ttk.Label(frame, text="Inverse Law: --")
        inverse_label.grid(row=7, column=0, columnspan=2)

        ratio_label = ttk.Label(frame, text="Overestimation Ratio (Exh/Inv): --")
        ratio_label.grid(row=8, column=0, columnspan=2, pady=(10,5))

        try:
            base_dir = os.path.dirname(__file__)
            image_path = os.path.join(base_dir, "search.png")
            help_img_raw = Image.open(image_path)
            help_img_raw = help_img_raw.resize((520, 370), Image.Resampling.LANCZOS)
            help_img = ImageTk.PhotoImage(help_img_raw)
            image_label = tk.Label(frame, image=help_img, bg="lightyellow")
            image_label.image = help_img
            image_label.grid(row=9, column=0, columnspan=2, pady=10)
        except Exception as e:
            error_label = ttk.Label(frame, text=f"Could not load image: {e}", foreground="red")
            error_label.grid(row=9, column=0, columnspan=2, pady=10)

        fig, ax = plt.subplots(figsize=(7, 5))
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.get_tk_widget().pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)

    # ---------------- Hit Probability GUI ----------------
    def create_hit_probability_gui(self, tab):
        def safe_float(entry_widget, name):
            val = entry_widget.get()
            if val.strip() == "":
                raise ValueError(f"{name} is required.")
            try:
                return float(val)
            except ValueError:
                raise ValueError(f"{name} must be a number.")

        def calculate_hit_probability():
            try:
                shape = shape_var.get()
                sigma = safe_float(entry_sigma, "σ (Standard Deviation)")
                if sigma <= 0:
                    raise ValueError("σ must be positive.")

                mu_x = safe_float(entry_mu_x, "μx (X-offset)")
                mu_y = safe_float(entry_mu_y, "μy (Y-offset)")

                fig.clear()
                ax = fig.add_subplot(111)

                if shape == "Circular":
                    R = safe_float(entry_radius, "Radius R")
                    if R <= 0:
                        raise ValueError("Radius R must be positive.")

                    r = np.sqrt(mu_x ** 2 + mu_y ** 2)
                    if r == 0:
                        ph = 1 - np.exp(-R ** 2 / (2 * sigma ** 2))
                    else:
                        ph = 1 - np.exp(-(R ** 2 + r ** 2) / (2 * sigma ** 2)) * np.i0(R * r / sigma ** 2)
                    ax.set_title("Circular Target")
                    ax.add_patch(plt.Circle((mu_x, mu_y), R, fill=False, color='blue'))

                elif shape == "Rectangular":
                    a = safe_float(entry_a, "Half-length a")
                    b = safe_float(entry_b, "Half-breadth b")
                    if a <= 0 or b <= 0:
                        raise ValueError("Half-length a and Half-breadth b must be positive.")

                    ph = (normal_cdf((a - mu_x) / sigma) - normal_cdf((-a - mu_x) / sigma)) * \
                         (normal_cdf((b - mu_y) / sigma) - normal_cdf((-b - mu_y) / sigma))
                    ax.set_title("Rectangular Target")
                    ax.add_patch(plt.Rectangle((mu_x - a, mu_y - b), 2 * a, 2 * b, fill=False, color='green'))

                elif shape == "Elliptical":
                    c = safe_float(entry_c, "Semi-major axis c")
                    d = safe_float(entry_d, "Semi-minor axis d")
                    if c <= 0 or d <= 0:
                        raise ValueError("Axes c and d must be positive.")

                    grid_size = 500
                    x = np.linspace(mu_x - c - 3 * sigma, mu_x + c + 3 * sigma, grid_size)
                    y = np.linspace(mu_y - d - 3 * sigma, mu_y + d + 3 * sigma, grid_size)
                    X, Y = np.meshgrid(x, y)
                    pdf = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((X - mu_x) ** 2 + (Y - mu_y) ** 2) / (2 * sigma ** 2))
                    mask = ((X - mu_x) / c) ** 2 + ((Y - mu_y) / d) ** 2 <= 1
                    ph = np.sum(pdf[mask]) * (x[1] - x[0]) * (y[1] - y[0])
                    ax.set_title("Elliptical Target")
                    ellipse = plt.matplotlib.patches.Ellipse((mu_x, mu_y), 2 * c, 2 * d, fill=False, color='red')
                    ax.add_patch(ellipse)

                ax.set_xlim(-10, 10)
                ax.set_ylim(-10, 10)
                ax.set_aspect('equal')
                ax.grid(True)
                canvas.draw()

                result_label.config(text=f"Hit Probability: {ph:.4f}", foreground="blue")

            except Exception as e:
                result_label.config(text=f"Error: {e}", foreground="red")

        def normal_cdf(z):
            return 0.5 * (1 + math.erf(z / math.sqrt(2)))

        def update_inputs(*args):
            for widget in [frame_circle, frame_rect, frame_ellipse]:
                widget.pack_forget()

            shape = shape_var.get()
            if shape == "Circular":
                frame_circle.pack()
            elif shape == "Rectangular":
                frame_rect.pack()
            elif shape == "Elliptical":
                frame_ellipse.pack()

        # GUI Setup
        window_width = 1100
        window_height = 600
        screen_width = tab.winfo_screenwidth()
        screen_height = tab.winfo_screenheight()
        x_cordinate = int((screen_width / 2) - (window_width / 2))
        y_cordinate = int((screen_height / 2) - (window_height / 2))
        tab.geometry = f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}"

        frame_help = ttk.Frame(tab, width=300)
        frame_help.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10)

        frame_input = ttk.Frame(tab)
        frame_input.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        frame_plot = ttk.Frame(tab)
        frame_plot.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=10)

        base_dir = os.path.dirname(__file__)
        image_path = os.path.join(base_dir, "hit.png")
        help_image = Image.open(image_path)
        help_image = help_image.resize((520, 630), Image.LANCZOS)
        help_photo = ImageTk.PhotoImage(help_image)
        label_image = ttk.Label(frame_help, image=help_photo)
        label_image.image = help_photo
        label_image.pack(fill=tk.BOTH)

        ttk.Label(frame_input, text="Target Shape").pack()
        shape_var = tk.StringVar()
        shape_combo = ttk.Combobox(frame_input, textvariable=shape_var,
                                   values=["Circular", "Rectangular", "Elliptical"], state="readonly")
        shape_combo.pack()
        shape_combo.bind("<<ComboboxSelected>>", update_inputs)
        shape_combo.current(0)

        ttk.Label(frame_input, text="σ (Standard Deviation):").pack()
        entry_sigma = ttk.Entry(frame_input)
        entry_sigma.pack()
        entry_sigma.insert(0,"2.0") #Default Standard Deviation

        ttk.Label(frame_input, text="μx (X-offset):").pack()
        entry_mu_x = ttk.Entry(frame_input)
        entry_mu_x.pack()
        entry_mu_x.insert(0,"0.0") #default X-offset

        ttk.Label(frame_input, text="μy (Y-offset):").pack()
        entry_mu_y = ttk.Entry(frame_input)
        entry_mu_y.pack()
        entry_mu_y.insert(0,"0.0") #default Y-offset

        frame_circle = ttk.Frame(frame_input)
        ttk.Label(frame_circle, text="Radius R:").pack()
        entry_radius = ttk.Entry(frame_circle)
        entry_radius.pack()
        entry_radius.insert(0,"4.0") #deafult radius

        frame_rect = ttk.Frame(frame_input)
        ttk.Label(frame_rect, text="Half-length a:").pack()
        entry_a = ttk.Entry(frame_rect)
        entry_a.pack()
        entry_a.insert(0,"3.545") #half-lenghth=3.545m -> full-length=7.09m
        ttk.Label(frame_rect, text="Half-breadth b:").pack()
        entry_b = ttk.Entry(frame_rect)
        entry_b.pack()
        entry_b.insert(0,"3.545") #half-lenghth=3.545m -> full-length=7.09m

        frame_ellipse = ttk.Frame(frame_input)
        ttk.Label(frame_ellipse, text="Semi-major axis c:").pack()
        entry_c = ttk.Entry(frame_ellipse)
        entry_c.pack()
        entry_c.insert(0,"5.0") #smi-major = 5m
        ttk.Label(frame_ellipse, text="Semi-minor axis d:").pack()
        entry_d = ttk.Entry(frame_ellipse)
        entry_d.pack()
        entry_d.insert(0,"3.0") #semi-minor = 3m

        frame_circle.pack()

        ttk.Button(frame_input, text="Calculate", command=calculate_hit_probability).pack(pady=10)
        result_label = ttk.Label(frame_input, text="Hit Probability: ", font=("Arial", 12))
        result_label.pack(pady=5)

        fig = plt.Figure(figsize=(5.5, 5), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=frame_plot)
        canvas.get_tk_widget().pack()


    #------------------Damage Assessment---------------
    def create_damage_assessment_gui(self, tab):
        from scipy.special import i0
        from scipy.integrate import quad
        import numpy as np
        from tkinter import StringVar

        def cookie_cutter_damage_integrand(r, a, n):
            def inner_integral(t):
                return np.exp(-(t ** 2) / 2) * i0(r * t)
            inner_result, _ = quad(inner_integral, 0, a)
            P_ar = np.exp(-(r ** 2) / 2) * inner_result
            P_ar = min(P_ar, 1.0)
            return 1 - (1 - P_ar) ** n

        def compute_cookie_cutter(R, a, n):
            def integrand(r):
                return cookie_cutter_damage_integrand(r, a, n) * r
            result, _ = quad(integrand, 0, R)
            return 1 - (2 / R ** 2) * result

        def gaussian_damage_integrand(r, alpha, n):
            P = np.exp(-alpha * r ** 2)
            return 1 - (1 - P) ** n

        def compute_gaussian(R, alpha, n):
            def integrand(r):
                return gaussian_damage_integrand(r, alpha, n) * r

            result, _ = quad(integrand, 0, R)
            return 1 - (2 / R ** 2) * result

        def cluster_damage_integrand(r, a, m):
            def P_single(r):
                val = np.exp(-((r / a) ** 2))
                return max(val, 1e-6)  # Avoid zero
            return 1 - (1 - P_single(r)) ** m

        def compute_cluster(R, a, m):
            def integrand(r):
                return cluster_damage_integrand(r, a, m) * r

            result, _ = quad(integrand, 0, R)
            return 1 - (2 / R ** 2) * result

        # Layout
        help_frame = ttk.Frame(tab, padding=5)
        help_frame.pack(side="left", fill="y")

        input_frame = ttk.Frame(tab, padding=10)
        input_frame.pack(side="left", fill="y")

        output_frame = ttk.Frame(tab, padding=10)
        output_frame.pack(side="right", fill="both", expand=True)

        try:
            base_dir = os.path.dirname(__file__)
            image_path = os.path.join(base_dir, "damage.png")
            image = Image.open(image_path)
            image = image.resize((450, 640), resample=Image.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            label = tk.Label(help_frame, image=photo)
            label.image = photo
            label.pack()
        except Exception as e:
            label = tk.Label(help_frame, text="Help image not found.\n\nError: " + str(e))
            label.pack()

        #input fields
        ttk.Label(input_frame, text="Damage Function:").pack()
        model_var = StringVar()
        model_dropdown = ttk.Combobox(
            input_frame, 
            textvariable=model_var,
            values=["Cookie-Cutter", "Gaussian", "Cluster Bombs"],
            state="readonly"
        )
        model_dropdown.pack()

        ttk.Label(input_frame, text=" Target Radius (m):").pack()
        entry_R = ttk.Entry(input_frame)
        entry_R.pack()

        ttk.Label(input_frame, text=" Lethal Radius (m):").pack()
        entry_a = ttk.Entry(input_frame)
        entry_a.pack()

        ttk.Label(input_frame, text=" CEP (m):").pack()
        entry_CEP = ttk.Entry(input_frame)
        entry_CEP.pack()

        ttk.Label(input_frame, text=" No. of Weapons (n):").pack()
        entry_n = ttk.Entry(input_frame)
        entry_n.pack()

        result_label = ttk.Label(input_frame, text="", foreground="blue")
        result_label.pack(pady=10)

        #preload default values
        def load_defaults(event=None):
            defaults = {
            "Cookie-Cutter":  ("3.14", "0.392", "127.4", "120"),
            "Gaussian":       ("3.53", "0.883", "84.95", "10"),
            "Cluster Bombs":  ("1.96", "0.147", "101.9", "200")
        }
            model = model_var.get()
            if model in defaults:
                r, a, cep, n = defaults[model]
                entry_R.delete(0, tk.END); entry_R.insert(0, r)
                entry_a.delete(0, tk.END); entry_a.insert(0, a)
                entry_CEP.delete(0, tk.END); entry_CEP.insert(0, cep)
                entry_n.delete(0, tk.END); entry_n.insert(0, n)

        model_dropdown.bind("<<ComboboxSelected>>", load_defaults)

        def calculate():
            try:
                R_m = float(entry_R.get())
                a_m = float(entry_a.get())
                CEP = float(entry_CEP.get())
                n = int(entry_n.get())
                model = model_dropdown.get()

                if R_m <= 0 or a_m <= 0 or CEP <= 0 or n <= 0:
                    raise ValueError("All inputs must be positive numbers.")

                sigma = CEP / 1.1774
                R = R_m / sigma
                a = a_m / sigma

                if model == "Cookie-Cutter":
                    E = compute_cookie_cutter(R, a, n)
                elif model == "Gaussian":
                    alpha = 1 / (a ** 2)
                    E = compute_gaussian(R, alpha, n)
                elif model == "Cluster Bombs":
                    m = n * 10
                    E = compute_cluster(R, a, m)
                else:
                    raise ValueError("Invalid model")

                result_label.config(text=f" Expected Damaged Area = {E * 100:.2f}%")

                for w in output_frame.winfo_children():
                    w.destroy()

                r_vals = np.linspace(0, R, 200)
                if model == "Cookie-Cutter":
                    p_vals = [cookie_cutter_damage_integrand(r, a, n) * 100 for r in r_vals]
                elif model == "Gaussian":
                    alpha = 1 / (a ** 2)
                    p_vals = [gaussian_damage_integrand(r, alpha, n) * 100 for r in r_vals]
                else:
                    m = n * 10
                    p_vals = [cluster_damage_integrand(r, a, m) * 100 for r in r_vals]

                fig, ax = plt.subplots(figsize=(5, 4))
                ax.plot(r_vals, p_vals, label=model, color="crimson")
                ax.set_xlabel("Normalized Distance r")
                ax.set_ylabel("Damage Probability (%)")
                ax.set_title("Damage vs Distance")
                ax.grid(True)
                ax.legend()

                canvas = FigureCanvasTkAgg(fig, master=output_frame)
                canvas.draw()
                canvas.get_tk_widget().pack()

            except Exception as e:
                result_label.config(text=f" Error: {e}")

        ttk.Button(input_frame, text=" Calculate", command=calculate).pack(pady=5)

        model_var.set("Cookie-Cutter")
        load_defaults()


    #--------------------Salvo & Pttern Firing----------------------------
    def create_salvo_pattern_gui(self, tab):
        from scipy.stats import norm

        def calculate_probabilities():
            try:
                a = float(entry_a.get())
                b = float(entry_b.get())
                sigma = float(entry_sigma.get())

                if a < 0:
                    raise ValueError("Spacing 'a' cannot be negative.")
                if b <= 0:
                    raise ValueError("Lethal radius 'b' must be positive.")
                if sigma <= 0:
                    raise ValueError("Aiming error σ must be positive.")
                
                # Salvo, pattern, independent
                pd_salvo = 2 * norm.cdf(b / sigma) - 1
                pd_pattern = 2 * (norm.cdf((a + b) / sigma) - norm.cdf((a - b) / sigma))
                pd_indep = 2 * pd_salvo - pd_salvo ** 2

                result_text.set(
                    f"Probability of Damage (Pd):\n\n"
                    f"Salvo Fire (2 bombs):         {pd_salvo:.4f}\n"
                    f"Pattern Fire (2 bombs stick): {pd_pattern:.4f}\n"
                    f"Independent Fire:             {pd_indep:.4f}\n"
                )

                ax.clear()
                labels = ['Salvo', 'Pattern', 'Independent']
                values = [pd_salvo, pd_pattern, pd_indep]
                colors=['skyblue', 'lightgreen', 'salmon']
                bars = ax.bar(labels, values, color=colors)
                  # Dynamic ylim with a little headroom (so labels/axes don't clip)
                top = max(1.0, max(values) * 1.08)
                ax.set_ylim(0.0, top)

            # offset relative to axis height
                offset = top * 0.04

            # Place labels: inside tall bars (val >= 0.9) with white text, above others with black text
                for bar, val in zip(bars, values):
                    x = bar.get_x() + bar.get_width() / 2.0
                    if val >= 0.90:                 # tall bar -> put label inside
                        y = val + offset
                        color = 'black'
                        va = 'center'
                    else:                           # smaller bar -> put label above
                        y = val + offset / 2.0
                        color = 'black'
                        va = 'bottom'
                    ax.text(x, y, f"{val:.2f}", ha='center', va=va, color=color, fontweight='bold')

                ax.set_ylabel("Probability")
                ax.set_title("Comparison of Firing Techniques")
                ax.grid(axis='y', linestyle=':', alpha=0.5)
                canvas.draw()
            except ValueError as ve:
                messagebox.showerror("Input Error", str(ve))

        # GUI Elements 
        frame_left = tk.Frame(tab, padx=10, pady=10)
        frame_left.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(frame_left, text="Inputs", font=('Arial', 12, 'bold')).grid(row=0, column=0, columnspan=2)

        tk.Label(frame_left, text="Spacing a (meters):").grid(row=1, column=0, sticky='w')
        entry_a = tk.Entry(frame_left)
        entry_a.grid(row=1, column=1)
        entry_a.insert(0, "1") #default spacing

        tk.Label(frame_left, text="Lethal Radius b (meters):").grid(row=2, column=0, sticky='w')
        entry_b = tk.Entry(frame_left)
        entry_b.grid(row=2, column=1)
        entry_b.insert(0, "1") #default lethal radius 

        tk.Label(frame_left, text="Aiming Error σ (meters):").grid(row=3, column=0, sticky='w')
        entry_sigma = tk.Entry(frame_left)
        entry_sigma.grid(row=3, column=1)
        entry_sigma.insert(0, "1") #default aiming error σ=1

        tk.Button(frame_left, text="Calculate", command=calculate_probabilities, bg="lightblue").grid(
            row=5, column=0, columnspan=2, pady=10)

        result_text = tk.StringVar()
        label_result = tk.Label(frame_left, textvariable=result_text, justify="left", font=("Courier", 10), fg="darkblue")
        label_result.grid(row=6, column=0, columnspan=2, sticky='w', pady=10)

        try:
            base_dir = os.path.dirname(__file__)
            image_path = os.path.join(base_dir, "salvopattern.png")
            image = Image.open(image_path)
            image = image.resize((500, 400))
            img_tk = ImageTk.PhotoImage(image)
            img_label = tk.Label(frame_left, image=img_tk)
            img_label.image = img_tk
            img_label.grid(row=7, column=0, columnspan=2, pady=5)
        except Exception as e:
            fallback = tk.Label(frame_left, text="Help image not found.\nPlease add 'salvopattern.png' to folder.", fg="red")
            fallback.grid(row=7, column=0, columnspan=2)

        frame_plot = tk.Frame(tab)
        frame_plot.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        fig, ax = plt.subplots(figsize=(14, 10))
        canvas = FigureCanvasTkAgg(fig, master=frame_plot)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # -------------- SINGLE vs MULTIPLE Aiming GUI Function --------------
    def create_single_vs_multiple_gui(self, tab):
        import os
        def calculate_coverage():
            try:
                c = float(entry_c.get())
                d = float(entry_d.get())
                q = int(entry_q.get())
                r_lethal = float(entry_lethal.get())
                cep = float(entry_cep.get())

                if c <= 0 or d <= 0:
                    raise ValueError("Target dimensions must be positive.")
                if q <= 0:
                    raise ValueError("Total shells must be a positive integer.")
                if r_lethal <= 0:
                    raise ValueError("Lethal radius must be positive.")
                if cep <= 0:
                    raise ValueError("CEP must be positive.")

                mode = mode_var.get()
                sigma = cep / 1.1774
                a = r_lethal / sigma

                if mode == "Single":
                    aiming_points = [(0.0, 0.0)]
                    q_list = [q]
                else:
                    aiming_points = [(-c / 2.0, 0.0), (c / 2.0, 0.0)]
                    shells_per_point = q // len(aiming_points)  # integer division
                    q_list = [shells_per_point] * len(aiming_points)

                num_points = 500
                x_vals = np.linspace(-c, c, num_points)
                y_vals = np.linspace(-d, d, num_points)
                dx = (2 * c) / num_points
                dy = (2 * d) / num_points

                max_corner_dist = max(
                    math.hypot(cx - x0, cy - y0)
                    for (x0, y0) in aiming_points
                    for cx in (-c, c)
                    for cy in (-d, d)
                )
                t_max = max_corner_dist / sigma
                try:
                    bessel_i0 = np.i0
                except AttributeError:
                    from scipy.special import i0 as bessel_i0
                
                def compute_p_grid(a, t_max, nt_t=500, n_int=800):
                    t_grid = np.linspace(0.0, t_max, nt_t)
                    u = np.linspace(0.0, a, n_int)
                    exp_u = np.exp(-u**2 / 2.0)
                    p_vals = np.zeros_like(t_grid)
                    for idx, t in enumerate(t_grid):
                        integrand = u * exp_u * bessel_i0(t * u)
                        if hasattr(np, "trapezoid"):
                            integral = np.trapezoid(integrand, u)  # NumPy >= 2.0
                        else:
                            integral = np.trapz(integrand, u)      # NumPy < 2.0
                        p_vals[idx] = math.exp(-t**2 / 2.0) * integral
                    return t_grid, p_vals

                t_grid, p_vals = compute_p_grid(a, t_max)
                def P_interp(t):
                    return np.interp(t, t_grid, p_vals)
            
                total_hits = 0.0
                for xi in x_vals:
                    for yi in y_vals:
                        prob_not_hit = 1.0
                        for (x0, y0), qj in zip(aiming_points, q_list):
                            tj = math.hypot(xi - x0, yi - y0) / sigma
                            p_single = P_interp(tj)
                            prob_not_hit *= (1 - p_single) ** qj
                        total_hits += (1 - prob_not_hit)

                area = 4 * c * d
                coverage_percent = (total_hits * dx * dy) / area * 100
                result_label.config(text=f"Coverage: {coverage_percent:.2f}%")

                ax.clear()
                ax.set_title("Target Area and Aiming Points")
                ax.set_xlim(-c, c)
                ax.set_ylim(-d, d)
                ax.set_aspect('equal')
                ax.set_xlabel("X (m)")
                ax.set_ylabel("Y (m)")

                rect_x = [-c, c, c, -c, -c]
                rect_y = [-d, -d, d, d, -d]
                ax.plot(rect_x, rect_y, linestyle='-', linewidth=1)
            
                for i, (x, y) in enumerate(aiming_points):
                    ax.plot(x, y, 'ro', markersize=8, markeredgecolor='black', markeredgewidth=1.5)
                    ax.text(x + 5, y + 5, f"P{i+1} ({x:.1f}, {y:.1f})", color='red')
                
                inset = min(c,d) * 0.05
                corners = [
                    (-c + inset, -d + inset),
                    (c - inset, -d + inset), 
                    (c - inset, d - inset), 
                    (-c + inset, d - inset)
                ]
                for j, (cx, cy) in enumerate(corners, start=1):
                    ax.plot(cx, cy, marker='s', markersize=7, markeredgewidth=2)
                    ax.text(cx + 5, cy + 5, f"C{j} ({cx:.0f},{cy:.0f})", fontsize=9, fontweight='bold', ha='center', va='center') 
                    if j in [1, 2]:  # bottom corners

                        ax.text(cx, cy - (d * 0.08), f"C{j}",
                            fontsize=9, fontweight='bold', ha='center', va='top')
                    else:  # top corners
                        ax.text(cx, cy + (d * 0.08), f"C{j}",
                            fontsize=9, fontweight='bold', ha='center', va='top')   
                
                offset_x = 15
                offset_y = 15
                # Assuming your corners are in the order: 
                # C1 (bottom-left), C2 (top-left), C3 (top-right), C4 (bottom-right)
                for i in range(len(corners)):
                    x, y = corners[i]
                    ax.plot(x, y, 'ro', markersize=8, markeredgewidth=1.5)

                # Decide position
                if i in [0, 3]:  # bottom corners
                    va = 'top'
                    y_text = y - offset_y
                else:            # top corners
                    va = 'top'
                    y_text = y + offset_y

                if i in [0, 1]:  # left corners
                    ha = 'right'
                    x_text = x - offset_x
                else:            # right corners
                    ha = 'left'
                    x_text = x + offset_x

                ax.annotate(f"C{i+1} ({x:.1f}, {y:.1f})",
                            (x, y),
                            xytext=(x_text, y_text),
                            textcoords='data',
                            ha=ha, va=va,
                            fontsize=10,
                            color='black',)
                canvas.draw()

            except ValueError as ve:
                messagebox.showerror("Invalid Input", str(ve))
            except Exception:
                messagebox.showerror("Error", "Please enter valid numeric values.")
        
        def toggle_shells_entry(*args):
            if mode_var.get() == "Multiple":
                entry_q.config(state="normal")  # Allow entry, but distribute later
            else:
                entry_q.config(state="normal")

        # Layout
        frame_left = tk.Frame(tab)
        frame_left.pack(side=tk.LEFT, padx=10, pady=10)

        frame_right = tk.Frame(tab)
        frame_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        labels = [
            "Half Width of Target (c in m):",
            "Half Height of Target (d in m):",
            "Total Shells Fired (q):",
            "Lethal Radius (m):",
            "CEP (m):"
        ]
        default_values = [375, 250, 60, 30, 100]
        entries = []
        for i, label_text in enumerate(labels):
            label = tk.Label(frame_left, text=label_text)
            label.grid(row=i, column=0, sticky="w")
            entry = tk.Entry(frame_left)
            entry.grid(row=i, column=1)
            entry.insert(0, str(default_values[i]))
            entries.append(entry)

        entry_c, entry_d, entry_q, entry_lethal, entry_cep = entries

        mode_var = tk.StringVar(value="Single")
        mode_label = tk.Label(frame_left, text="Firing Mode:")
        mode_label.grid(row=5, column=0, sticky="w")
        single_radio = ttk.Radiobutton(frame_left, text="Single Aiming Point", variable=mode_var, value="Single")
        single_radio.grid(row=5, column=1, sticky="w")
        multi_radio = ttk.Radiobutton(frame_left, text="Multiple Aiming Points", variable=mode_var, value="Multiple")
        multi_radio.grid(row=6, column=1, sticky="w")

        calc_button = tk.Button(frame_left, text="Calculate Coverage", command=calculate_coverage)
        calc_button.grid(row=7, column=0, columnspan=2, pady=10)

        result_label = tk.Label(frame_left, text="Coverage: ")
        result_label.grid(row=8, column=0, columnspan=2)

        try:
            base_dir = os.path.dirname(__file__)
            image_path = os.path.join(base_dir, "singlevsmultiple.png")
            if os.path.exists(image_path):
                help_img = Image.open(image_path)
                help_img = help_img.resize((550, 390))
                help_photo = ImageTk.PhotoImage(help_img)
                help_label = tk.Label(frame_left, image=help_photo)
                help_label.image = help_photo
                help_label.grid(row=9, column=0, columnspan=2, pady=10)
            else:
                tk.Label(frame_left, text="Help image not found!").grid(row=9, column=0, columnspan=2)
        except Exception:
            tk.Label(frame_left, text="Error loading help image").grid(row=9, column=0, columnspan=2)

        fig, ax = plt.subplots(figsize=(5, 5))
        canvas = FigureCanvasTkAgg(fig, master=frame_right)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


        # -------------- SHOOTING TACTICS GUI Function --------------
    def create_shooting_tactics_gui(self, tab):
        from math import comb
        def calculate_expected_kills():
            try:
                k_str, n_str, p_str = entry_k.get(), entry_n.get(), entry_p.get()

                if not k_str.isdigit() or not n_str.isdigit():
                    raise ValueError("Total targets and rounds must be integers.")

                k = int(k_str)
                n = int(n_str)
                p = float(p_str)

                if k <= 0 or n < 0 or not (0 <= p <= 1):
                    raise ValueError("Enter: k > 0, n ≥ 0, 0 ≤ p ≤ 1")

                if k > 1000:
                    raise ValueError("Please enter k ≤ 1000 to allow proper plotting.")

                expected_kills = k * (1 - ((1 - p / k) ** n))
                wasted_rounds = n - expected_kills
                result_label.config(
                    text=f"Expected Kills: {expected_kills:.2f}\n"
                        f"Wasted Rounds:{n:.2f}-{ expected_kills:.2f} = {wasted_rounds:.2f}"
                        )

                def Pn(m):
                    total = 0
                    for j in range(m + 1):
                        total += ((-1) ** j) * comb(m, j) * (((k - m +j) / k) ** n)
                    return comb(k, m) * total

                x_vals = np.arange(0, k + 1)
                y_vals = [Pn(m) for m in x_vals]

                ax.clear()
                ax.bar(x_vals, y_vals)
                ax.set_title("Probability Distribution of Kills (Pn(m))")
                ax.set_xlabel("Number of Kills (m)")
                ax.set_ylabel("Probability")
                ax.set_ylim(0, 1)  # keep probability scale correct
                canvas.draw()

            except ValueError as e:
                messagebox.showerror("Invalid input", str(e))

        frame_left = tk.Frame(tab)
        frame_left.pack(side=tk.LEFT, padx=10, pady=10)

        frame_right = tk.Frame(tab)
        frame_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        labels = ["Total Targets (k):", "Total Rounds Fired (n):", "Single Shot Kill Probability (p):"]
        entries = []
        for i, label_text in enumerate(labels):
            label = tk.Label(frame_left, text=label_text)
            label.grid(row=i, column=0, sticky="w")
            entry = tk.Entry(frame_left)
            entry.grid(row=i, column=1)
            entries.append(entry)

        entry_k, entry_n, entry_p = entries
        # Set default values (k=10, n=10, p=1)
        entry_k.insert(0, "10")
        entry_n.insert(0, "10")
        entry_p.insert(0, "1")

        calc_button = tk.Button(frame_left, text="Calculate Expected Kills", command=calculate_expected_kills)
        calc_button.grid(row=4, column=0, columnspan=2, pady=10)

        result_label = tk.Label(frame_left, text="Expected Kills: ")
        result_label.grid(row=5, column=0, columnspan=2)

        try:
            base_dir = os.path.dirname(__file__)
            image_path = os.path.join(base_dir, "shooting.png")
            help_image_raw = Image.open(image_path)
            help_image_resized = help_image_raw.resize((400, 450), Image.LANCZOS)
            help_image = ImageTk.PhotoImage(help_image_resized)
            help_label = tk.Label(frame_left, image=help_image)
            help_label.image = help_image
            help_label.grid(row=6, column=0, columnspan=2, pady=10)
        except Exception:
            error_label = tk.Label(frame_left, text="Help image not found or cannot be loaded.", fg="red")
            error_label.grid(row=6, column=0, columnspan=2, pady=10)

        fig, ax = plt.subplots(figsize=(5, 4))
        canvas = FigureCanvasTkAgg(fig, master=frame_right)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

  

# ---------------- Launch App ----------------
if __name__ == '__main__':
    root = tk.Tk()
    app = CombinedApp(root)
    root.mainloop()

