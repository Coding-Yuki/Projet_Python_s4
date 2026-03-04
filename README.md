# Mask Detection Pro G11

A real-time face mask detection application using TensorFlow Lite and MediaPipe. Featuring a futuristic Sci-Fi HUD and high-performance inference.

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.9+** installed on your system.

### 2. Setup Environment
Open your terminal in the project folder and run:

```bash
# Create a virtual environment
python -m venv venv

# Activate it (Windows)
.\venv\Scripts\activate

# Activate it (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Application
To launch the real-time detector:

```bash
python main.py run
```

---

## 🎮 Controls
| Key | Action |
|-----|--------|
| **Q** | Quit Application |
| **G** | Toggle Grad-CAM Visualization |
| **S** | Save Screenshot |
| **SPACE** | Pause / Resume Detection |

## 🛠️ Project Structure
- `main.py`: Entry point for all commands (train, evaluate, export, run).
- `src/`: Core logic for training and evaluation.
- `models/`: Pre-trained model files (Keras and TFLite).
- `app/`: HUD utilities and the real-time application script.

## 📄 License
This project is for educational purposes.
