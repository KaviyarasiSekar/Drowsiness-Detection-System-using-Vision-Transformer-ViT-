import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from torchvision import transforms
from transformers import ViTForImageClassification, ViTConfig
import os

class DrowsinessDetectionGUI:
    def __init__(self, master):
        self.master = master
        master.title("Drowsiness Detection")

        # Set up the device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using Metal GPU acceleration")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        self.model = self.load_model()

        self.select_button = tk.Button(master, text="Select Image", command=self.select_image)
        self.select_button.pack(pady=10)

        self.stop_button = tk.Button(master, text="Stop", command=self.stop_processing)
        self.stop_button.pack(pady=10)

        self.image_label = tk.Label(master)
        self.image_label.pack(pady=10)

        self.result_label = tk.Label(master, text="")
        self.result_label.pack(pady=10)

        self.stop_label = tk.Label(master, text="", fg="red")
        self.stop_label.pack(pady=10)

        self.running = True

    def load_model(self):
        config = ViTConfig.from_pretrained("google/vit-base-patch16-224", num_labels=2)
        config.image_size = 160
        model = ViTForImageClassification(config)
        model.load_state_dict(torch.load('drowsiness_model.pth', map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def select_image(self):
        if not self.running:
            print("Process stopped. Please restart the application.")
            return

        file_path = filedialog.askopenfilename()
        if file_path:
            self.process_image(file_path)

    def process_image(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            result, confidence = self.predict_drowsiness(image)
            self.display_result(image, result, confidence, image_path)
        except Exception as e:
            print(f"Error processing image: {str(e)}")

    def predict_drowsiness(self, image):
        transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor).logits
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            not_drowsy_prob, drowsy_prob = probabilities[0]

            if not_drowsy_prob > drowsy_prob:
                result = "Not Drowsy"
                confidence = not_drowsy_prob.item()
            else:
                result = "Drowsy"
                confidence = drowsy_prob.item()

        return result, confidence * 100

    def display_result(self, image, result, confidence, image_path):
        image.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(image)

        self.image_label.config(image=photo)
        self.image_label.image = photo

        result_text = f"Image: {os.path.basename(image_path)}\nPrediction: {result}\nConfidence: {confidence:.2f}%"
        self.result_label.config(text=result_text)

    def stop_processing(self):
        self.running = False
        self.stop_label.config(text="Process stopped. Please close the application.")
        print("Process stopped. Please close the application.")

def main():
    root = tk.Tk()
    app = DrowsinessDetectionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()