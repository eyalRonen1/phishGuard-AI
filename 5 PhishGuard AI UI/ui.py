import tkinter as tk
from tkinter import messagebox, scrolledtext, filedialog, ttk
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import json
from datetime import datetime
from PIL import Image, ImageTk
import csv
import os
from email import policy
from email.parser import BytesParser

# Load model and tokenizer
model_path = "./models/fine_tuned_phishing_model"
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)


def predict_with_probability(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        phishing_prob = probabilities[0][1].item()
        safe_prob = probabilities[0][0].item()
    return phishing_prob, safe_prob


def rounded_rectangle(canvas, x1, y1, x2, y2, radius=25, **kwargs):
    points = [x1 + radius, y1,
              x1 + radius, y1,
              x2 - radius, y1,
              x2 - radius, y1,
              x2, y1,
              x2, y1 + radius,
              x2, y1 + radius,
              x2, y2 - radius,
              x2, y2 - radius,
              x2, y2,
              x2 - radius, y2,
              x2 - radius, y2,
              x1 + radius, y2,
              x1 + radius, y2,
              x1, y2,
              x1, y2 - radius,
              x1, y2 - radius,
              x1, y1 + radius,
              x1, y1 + radius,
              x1, y1]
    return canvas.create_polygon(points, **kwargs, smooth=True)


class PhishingDetectionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PhishGuard AI")
        self.geometry("800x680")
        self.configure(bg='#0b1225')
        self.create_toolbar()
        self.create_widgets()
        self.resizable(False, False)

        # Center the window on the screen
        self.update_idletasks()  # Ensure all geometry is calculated
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        window_width = self.winfo_width()
        window_height = self.winfo_height()
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        self.geometry(f"{window_width}x{window_height}+{x}+{y}")

    def create_toolbar(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Toolbar.TButton', background='#2C3E50', foreground='white', padding=5)
        style.map('Toolbar.TButton', background=[('active', '#34495E')])

        toolbar = ttk.Frame(self, style='Toolbar.TFrame')
        toolbar.pack(side=tk.TOP, fill=tk.X)

        # Load button
        load_button = ttk.Button(toolbar, text="Load", command=self.load_text, style='Toolbar.TButton')
        load_button.pack(side=tk.LEFT, padx=2, pady=2)

        # Save button
        save_button = ttk.Button(toolbar, text="Save", command=self.save_text, style='Toolbar.TButton')
        save_button.pack(side=tk.LEFT, padx=2, pady=2)

        # Clear button
        clear_button = ttk.Button(toolbar, text="Clear", command=self.clear_text, style='Toolbar.TButton')
        clear_button.pack(side=tk.LEFT, padx=2, pady=2)

        # History button
        history_button = ttk.Button(toolbar, text="History", command=self.show_history, style='Toolbar.TButton')
        history_button.pack(side=tk.LEFT, padx=2, pady=2)

        # Help button
        help_button = ttk.Button(toolbar, text="Help", command=self.show_help, style='Toolbar.TButton')
        help_button.pack(side=tk.LEFT, padx=2, pady=2)

        # Risk Levels button
        risk_levels_button = ttk.Button(toolbar, text="Risk Levels", command=self.show_risk_levels,
                                        style='Toolbar.TButton')
        risk_levels_button.pack(side=tk.LEFT, padx=2, pady=2)

        # Exit button
        exit_button = ttk.Button(toolbar, text="Exit", command=self.quit, style='Toolbar.TButton')
        exit_button.pack(side=tk.RIGHT, padx=2, pady=2)

    def create_widgets(self):
        self.canvas = tk.Canvas(self, bg='#0b1225', highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # Improved header
        self.canvas.create_rectangle(0, 0, 800, 70, fill="#4CAF50", outline="")
        self.canvas.create_text(400, 35, text="Phishing Detection", font=("Arial", 28, "bold"), fill="white")

        # Image
        self.load_image(400, 150, 500, 150)

        # Text input area
        rounded_rectangle(self.canvas, 50, 220, 750, 420, radius=20, fill="#ffffff", outline="#cccccc")
        self.text_input = scrolledtext.ScrolledText(self.canvas, wrap=tk.WORD, font=("Arial", 12), bd=0)
        self.canvas.create_window(400, 320, window=self.text_input, width=680, height=180)

        # Analyze button with rounded edges
        analyze_button = tk.Button(self.canvas, text="Analyze Email", command=self.analyze_email,
                                   bg="#4CAF50", fg="white", font=("Arial", 14, "bold"),
                                   relief="flat", padx=20, pady=10, borderwidth=0, highlightthickness=0)

        # Create a rounded rectangle behind the button
        button_width = 200
        button_height = 50
        rounded_rect = rounded_rectangle(self.canvas, 400 - button_width / 2, 470 - button_height / 2,
                                         400 + button_width / 2, 470 + button_height / 2,
                                         radius=25, fill="#4CAF50", outline="")

        # Place the button on top of the rounded rectangle
        button_window = self.canvas.create_window(400, 470, window=analyze_button)
        self.canvas.tag_raise(button_window, rounded_rect)

        # Result display setup
        self.result_frame = tk.Frame(self.canvas, bg='#0b1225', bd=0)
        self.canvas.create_window(400, 570, window=self.result_frame, width=700, height=120)

        # Create styled widgets for result display
        self.probability_frame = tk.Frame(self.result_frame, bg='#0b1225')
        self.probability_frame.pack(pady=5)

        self.phishing_prob_label = tk.Label(self.probability_frame, text="", bg='#0b1225', fg="white",
                                            font=("Arial", 12, "bold"))
        self.phishing_prob_label.pack(side=tk.LEFT, padx=10)

        self.safe_prob_label = tk.Label(self.probability_frame, text="", bg='#0b1225', fg="white",
                                        font=("Arial", 12, "bold"))
        self.safe_prob_label.pack(side=tk.LEFT, padx=10)

        self.risk_label = tk.Label(self.result_frame, text="", bg='#0b1225', fg="white", font=("Arial", 14, "bold"))
        self.risk_label.pack(pady=5)

        self.advice_label = tk.Label(self.result_frame, text="", bg='#0b1225', fg="white", font=("Arial", 12),
                                     wraplength=680, justify="center")
        self.advice_label.pack(pady=5)

    def load_image(self, x, y, width, height):
        image = Image.open("logo2.jpg")
        image = image.resize((width, height), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(x, y, image=self.photo)

    def load_text(self):
        # Clear previous recommendations and text box before loading a new file
        self.phishing_prob_label.config(text="")
        self.safe_prob_label.config(text="")
        self.risk_label.config(text="")
        self.advice_label.config(text="")
        self.text_input.delete(1.0, tk.END)  # Clear the text input box

        # Allow selection of .msg, .eml, and .txt files
        file_paths = filedialog.askopenfilenames(
            filetypes=[("Email and Text files", "*.msg *.eml *.txt"), ("All files", "*.*")]
        )

        if not file_paths:
            return  # Exit function if no files are selected

        results = []  # Initialize an empty list to store results for each file

        for file_path in file_paths:
            if file_path.endswith(".msg"):
                # Handle .msg file
                with open(file_path, 'r', errors='ignore') as file:
                    content = file.read()

            elif file_path.endswith(".eml"):
                # Handle .eml file
                with open(file_path, 'rb') as file:
                    msg = BytesParser(policy=policy.default).parse(file)
                    content = msg.get_body(preferencelist=('plain')).get_content() if msg.get_body(
                        preferencelist=('plain')) else ''

            elif file_path.endswith(".txt"):
                # Handle .txt file
                with open(file_path, 'r', errors='ignore') as file:
                    content = file.read()

            else:
                continue  # Skip unsupported file types

            # Analyze content and get phishing and safe probabilities
            phishing_prob, safe_prob = predict_with_probability(content)

            # Determine risk level and advice based on phishing probability
            if phishing_prob > 0.8:
                risk_level = "Critical Risk"
                color = "#FF0000"  # Red
                advice = "Extremely high risk of phishing! Do not interact in any way."
            elif phishing_prob > 0.6:
                risk_level = "High Risk"
                color = "#FFA500"  # Orange
                advice = "High risk. Verify sender identity through trusted channels."
            elif phishing_prob > 0.4:
                risk_level = "Moderate Risk"
                color = "#FFFF00"  # Yellow
                advice = "Moderate risk. Verify sender identity cautiously."
            elif phishing_prob > 0.2:
                risk_level = "Low Risk"
                color = "#90EE90"  # Light Green
                advice = "Low risk, but stay vigilant."
            else:
                risk_level = "Minimal Risk"
                color = "#00FF00"  # Green
                advice = "Minimal risk detected. Follow standard security practices."

            # Display results if only one file is selected
            if len(file_paths) == 1:
                self.phishing_prob_label.config(text=f"Phishing Probability: {phishing_prob * 100:.2f}%")
                self.safe_prob_label.config(text=f"Safe Probability: {safe_prob * 100:.2f}%")
                self.risk_label.config(text=f"Risk Level: {risk_level}", fg=color)
                self.advice_label.config(text=advice)

            # Append analysis result to results list for CSV saving
            results.append({
                "File Name": os.path.basename(file_path),  # Extract only the file name
                "Phishing Probability": phishing_prob * 100,
                "Safe Probability": safe_prob * 100,
                "Risk Level": risk_level
            })

        # Save to CSV if multiple files are selected
        if len(file_paths) > 1 and results:
            csv_file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])

            if csv_file_path:
                with open(csv_file_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                    fieldnames = ["File Name", "Phishing Probability", "Safe Probability", "Risk Level"]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                    writer.writeheader()
                    for result in results:
                        writer.writerow(result)

                messagebox.showinfo("Success", f"Phishing analysis saved to {csv_file_path}")

    def save_text(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                 filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file_path:
            content = self.text_input.get(1.0, tk.END)
            with open(file_path, 'w') as file:
                file.write(content)

    def clear_text(self):
        self.text_input.delete(1.0, tk.END)
        self.phishing_prob_label.config(text="")
        self.safe_prob_label.config(text="")
        self.risk_label.config(text="")
        self.advice_label.config(text="")

    def show_history(self):
        history_window = tk.Toplevel(self)
        history_window.title("Analysis History")
        history_window.geometry("600x400")

        history_text = scrolledtext.ScrolledText(history_window, wrap=tk.WORD, font=("Arial", 10))
        history_text.pack(expand=True, fill='both')

        try:
            with open("analysis_log.json", "r") as f:
                log = json.load(f)
                for entry in log:
                    history_text.insert(tk.END, f"Timestamp: {entry['timestamp']}\n")
                    history_text.insert(tk.END, f"Phishing Probability: {entry['phishing_prob']:.2%}\n")
                    history_text.insert(tk.END, f"Safe Probability: {entry['safe_prob']:.2%}\n")
                    history_text.insert(tk.END, f"Email Text: {entry['email_text'][:100]}...\n\n")
        except FileNotFoundError:
            history_text.insert(tk.END, "No analysis history found.")

    def show_help(self):
        help_window = tk.Toplevel(self)
        help_window.title("PhishGuard AI - Help")
        help_window.geometry("600x500")
        help_window.resizable(False, False)

        style = ttk.Style()
        style.configure("Help.TFrame", background="#f0f0f0")
        style.configure("Help.TLabel", background="#f0f0f0", font=("Arial", 12))
        style.configure("HelpTitle.TLabel", background="#f0f0f0", font=("Arial", 16, "bold"))

        main_frame = ttk.Frame(help_window, style="Help.TFrame", padding="20 20 20 20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        title_label = ttk.Label(main_frame, text="Welcome to PhishGuard AI", style="HelpTitle.TLabel")
        title_label.pack(pady=(0, 20))

        text_widget = tk.Text(main_frame, wrap=tk.WORD, font=("Arial", 11), bg="#f0f0f0", relief="flat")
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.tag_configure("bold", font=("Arial", 11, "bold"))

        help_text = """
    PhishGuard AI is an advanced email analysis tool designed to detect and prevent phishing attempts. Here's how it works:

    1. Input: Enter the email text you want to analyze into the main text area.

    2. Analysis: Click the "Analyze Email" button to initiate the AI-powered analysis.

    3. AI Processing: Our state-of-the-art AI model, based on DistilBERT architecture, processes the email content.

    4. Risk Assessment: The software calculates the probability of the email being a phishing attempt.

    Remember: While PhishGuard AI is a powerful tool, always exercise caution with suspicious emails and consult with IT security professionals when in doubt.

    © 2024 PhishGuard AI. All Rights Reserved.
    Developed by Eyal Ronen & Gal Erez
    Version 1.0 (16/09/2024)
    """

        text_widget.insert(tk.END, help_text)
        text_widget.config(state=tk.DISABLED)  # Make the text read-only

        # Add a close button
        close_button = ttk.Button(main_frame, text="Close", command=help_window.destroy)
        close_button.pack(pady=(20, 0))

        # Center the window on the screen
        help_window.update_idletasks()
        width = help_window.winfo_width()
        height = help_window.winfo_height()
        x = (help_window.winfo_screenwidth() // 2) - (width // 2)
        y = (help_window.winfo_screenheight() // 2) - (height // 2)
        help_window.geometry('{}x{}+{}+{}'.format(width, height, x, y))

    def show_risk_levels(self):
        # Create a new window for displaying risk levels and recommendations
        risk_window = tk.Toplevel(self)
        risk_window.title("Risk Levels and Recommendations")
        risk_window.geometry("700x600")  # Set width and height for better layout
        risk_window.resizable(False, False)

        # Center the risk levels window on the screen
        risk_window.update_idletasks()  # Ensure all geometry is calculated
        screen_width = risk_window.winfo_screenwidth()
        screen_height = risk_window.winfo_screenheight()
        window_width = risk_window.winfo_width()
        window_height = risk_window.winfo_height()
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        risk_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # Add title label
        tk.Label(risk_window, text="Risk Levels and Recommendations", font=("Arial", 16, "bold")).pack(pady=10)

        # Define risk levels, thresholds, recommendations, and colors
        risk_info = [
            (
                "Minimal Risk", "Phishing Probability <= 20%",
                "Minimal risk detected. Follow standard security practices.",
                "#00FF00"),
            ("Low Risk", "Phishing Probability > 20%", "Low risk, but stay vigilant.", "#90EE90"),
            ("Moderate Risk", "Phishing Probability > 40%", "Moderate risk. Verify sender identity cautiously.",
             "#FFD700"),
            ("High Risk", "Phishing Probability > 60%", "High risk. Verify sender identity through trusted channels.",
             "#FFA500"),
            ("Critical Risk", "Phishing Probability > 80%",
             "Extremely high risk of phishing! Do not interact in any way.", "#FF0000")
        ]

        # Display each risk level with its color and recommendation
        for level, probability, recommendation, color in risk_info:
            tk.Label(risk_window, text=f"{level}", font=("Arial", 14, "bold"), fg=color).pack(anchor="w", padx=20,
                                                                                              pady=5)
            tk.Label(risk_window, text=f"Probability: {probability}", font=("Arial", 12)).pack(anchor="w", padx=40)
            tk.Label(risk_window, text=f"Recommendation: {recommendation}", font=("Arial", 12), wraplength=550).pack(
                anchor="w", padx=40, pady=(0, 10))

        # Close button
        close_button = ttk.Button(risk_window, text="Close", command=risk_window.destroy)
        close_button.pack(pady=15)

    def analyze_email(self):
        # Clear previous recommendations and analysis results
        self.phishing_prob_label.config(text="")
        self.safe_prob_label.config(text="")
        self.risk_label.config(text="")
        self.advice_label.config(text="")

        email_text = self.text_input.get("1.0", tk.END).strip()
        if not email_text:
            messagebox.showwarning("Input Error", "Please enter some text.")
            return

        phishing_prob, safe_prob = predict_with_probability(email_text)

        # Update probability labels
        self.phishing_prob_label.config(text=f"Phishing Probability: {phishing_prob * 100:.2f}%")
        self.safe_prob_label.config(text=f"Safe Probability: {safe_prob * 100:.2f}%")

        # Determine risk level, set color, and format `risk_label` as "Risk Level: _"
        if phishing_prob > 0.8:
            risk = "Critical Risk"
            color = "#FF0000"  # Red
            advice = "Extremely high risk of phishing! Do not interact in any way."
        elif phishing_prob > 0.6:
            risk = "High Risk"
            color = "#FFA500"  # Orange
            advice = "High risk. Verify sender identity through trusted channels."
        elif phishing_prob > 0.4:
            risk = "Moderate Risk"
            color = "#FFD700"  # Yellow
            advice = "Moderate risk. Verify sender identity cautiously."
        elif phishing_prob > 0.2:
            risk = "Low Risk"
            color = "#90EE90"  # Light Green
            advice = "Low risk, but stay vigilant."
        else:
            risk = "Minimal Risk"
            color = "#00FF00"  # Green
            advice = "Minimal risk. Follow standard security practices."

        # Update risk label to "Risk Level: _" and advice label with detailed advice
        self.risk_label.config(text=f"Risk Level: {risk}", fg=color)
        self.advice_label.config(text=advice)

        self.log_result(email_text, phishing_prob, safe_prob)

    def log_result(self, email_text, phishing_prob, safe_prob):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "email_text": email_text,
            "phishing_prob": phishing_prob,
            "safe_prob": safe_prob
        }
        try:
            with open("analysis_log.json", "r+") as f:
                log = json.load(f)
                log.append(log_entry)
                f.seek(0)
                json.dump(log, f, indent=2)
        except FileNotFoundError:
            with open("analysis_log.json", "w") as f:
                json.dump([log_entry], f, indent=2)


if __name__ == "__main__":
    app = PhishingDetectionApp()
    app.mainloop()