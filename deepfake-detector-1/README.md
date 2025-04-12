# Deepfake Detector

This project implements a deepfake detector using Convolutional Neural Networks (CNN) with TensorFlow. The model is designed to identify deepfake videos by analyzing the visual content.

## Project Structure

```
deepfake-detector
├── src
│   ├── data
│   │   ├── __init__.py
│   │   └── dataset.py
│   ├── models
│   │   ├── __init__.py
│   │   └── cnn_model.py
│   ├── training
│   │   └── train.py
│   └── utils
│       └── __init__.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd deepfake-detector
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To train the deepfake detector model, run the following command:

```
python src/training/train.py
```

This will load the dataset, initialize the CNN model, and start the training process. The trained model will be saved as `best_model.keras`.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
