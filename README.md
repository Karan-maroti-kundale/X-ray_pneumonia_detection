# 🖌️ Real-time Image to Sketch Animation

Convert images into **realistic animated pencil sketches** in real-time using **Python**, **OpenCV**, and **NumPy**. This project demonstrates advanced computer vision techniques for edge detection, contour generation, and animated stroke rendering, making it perfect for **creative, educational, and tech prototyping applications**.

---

## 🌟 Features

- **Real-time sketch animation**: Transform images into dynamic pencil sketches.  
- **Pencil shading effect**: Smooth and realistic shading for a hand-drawn look.  
- **Stroke-order animation**: Animates contours from top-to-bottom for natural drawing.  
- **GIF output**: Save the animation as a downloadable `.gif` file.  
- **Lightweight & efficient**: Works without neural networks (CNNs), making it fast and easy to deploy.

---

## 🔧 Tech Stack

- **Python 🐍**  
- **OpenCV 📸**  
- **NumPy 🔢**  
- **Pillow 🖼️**  
- **Imageio**  
- **Streamlit ⏱️** (for live web deployment)

---

## ⚡ Why OpenCV Instead of CNN

- ⚡ **Lightweight & fast**: Real-time processing without heavy model training.  
- 🛠️ **Full control**: Direct access to edges, contours, and shading for precise sketch effects.  
- 💻 **Easy deployment**: Runs smoothly on cloud platforms without GPU dependency.

---

## 🌐 Impact & Relevance

- **Creative industries**: Generate stylized content for marketing, social media, or digital art.  
- **Education**: Teach computer vision concepts interactively with visual tools.  
- **Tech companies**: Prototype AR/VR filters, mobile apps, and design tools.  

**🔥 Big Tech Relevance:**  
Companies like **Google, Adobe, NVIDIA, and Meta** are exploring real-time image processing, AI-assisted design, and animation technologies. Projects like this showcase **hands-on skills bridging creativity and technology**.

---

## 📁 Demo

- **Input Image:** `Demo_input.jpg`  
- **Output Sketch:** `Demo_output.jpg`  
- **Animation GIF:** `sketch_animation.gif`  

![Demo GIF](Demo_output.jpg) <!-- Replace with actual GIF if available -->

---

## 🚀 How to Run Locally

1. **Clone the repository**
```bash
git clone https://github.com/Karan-maroti-kundale/Real-time-image-to-sketch-Animation.git
cd Real-time-image-to-sketch-Animation
Create a virtual environment

bash
Copy code
python -m venv venv
# On Linux/Mac
source venv/bin/activate
# On Windows
venv\Scripts\activate
Install dependencies

bash
Copy code
pip install -r requirements.txt
Run the Streamlit app

bash
Copy code
streamlit run sketch_animation.py
🌐 Live Demo
Try it online: Real-time Image to Sketch Animation

📥 Download
You can download the animated GIF directly from the app after processing your image.`
