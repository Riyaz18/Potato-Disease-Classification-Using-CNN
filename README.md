# Potato-Disease-Classification-Using-CNN
This repository implements a Convolutional Neural Network (CNN) to classify potato leaf diseases like early blight, late blight, or healthy leaves. It includes data preprocessing, model training, and evaluation. Aimed at aiding farmers with quick disease detection for better crop management. Contributions are welcome!

# Potato Disease Classification

This repository contains a complete solution for classifying potato leaf diseases using a Convolutional Neural Network (CNN). The project is organized into several components to facilitate seamless development, training, and deployment.

## Project Structure

```
potato-disease/
├── .idea/           # IDE configuration files
├── __pycache__/     # Cached Python files
├── api/             # Backend API for model inference
├── frontend/        # Web interface for disease classification
├── mobile-app/      # Mobile application for real-time predictions
├── saved_models/    # Saved model checkpoints
├── training/        # Scripts for training the CNN model
├── model/           # Pre-trained and custom models
├── models/          # Additional model architectures and utilities
├── potatoes.h5      # Final trained model file
```

## Key Components

### 1. **API**
The `api` folder contains the backend code for serving the trained model. It uses a lightweight framework (e.g., Flask or FastAPI) to handle requests and return predictions.

### 2. **Frontend**
The `frontend` folder includes a web-based user interface for uploading images of potato leaves and receiving disease predictions.

### 3. **Mobile App**
The `mobile-app` folder provides a mobile application for farmers to classify diseases on-the-go using the trained model.

### 4. **Training**
The `training` folder contains Python scripts for training the CNN model. This includes preprocessing, data augmentation, and model evaluation.

### 5. **Models**
- `model`: Contains the main pre-trained or custom-trained model.
- `models`: Includes additional architectures or experimental models.
- `potatoes.h5`: The final trained model file ready for deployment.

### 6. **Saved Models**
The `saved_models` folder stores checkpoints of the training process for reuse and fine-tuning.

## Getting Started

### Prerequisites
- Python 3.x
- Required libraries (listed in `requirements.txt`)
- A compatible IDE (optional)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/potato-disease.git
   cd potato-disease
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Navigate to the desired component (e.g., `training`, `api`, etc.) and follow the instructions in its README.

### Usage
- **Training**: Run the training scripts in the `training` folder to train or fine-tune the model.
- **Inference**: Use the `api` or `frontend` to predict potato diseases using images.
- **Deployment**: Deploy the solution on the web or mobile platform for end-user accessibility.

## Contributing
Contributions are welcome! Feel free to submit issues, suggest improvements, or create pull requests.

## License
This project is licensed under the [MIT License](LICENSE).
