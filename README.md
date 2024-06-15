# Brain Tumor Segmentation with U-Net

This project uses a U-Net model to perform brain tumor segmentation on MRI images. The app is built using Streamlit, allowing users to upload MRI images, process them through the model, and visualize the segmented tumor.

## Features

- Upload MRI images in `.tif` format.
- Uses a pre-trained U-Net model to predict and segment brain tumors.
- Displays the original and predicted images side by side for comparison.

## Model Architecture

The U-Net model used in this project consists of an encoder-decoder architecture with skip connections to capture both the spatial and contextual information in the MRI images. It includes:

- Encoder: Convolutional layers with ReLU activation, followed by max-pooling layers.
- Decoder: Up-sampling layers followed by convolutional layers and concatenation with corresponding encoder layers.
- Dropout layers for regularization.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/brain-tumor-detection.git
    cd brain-tumor-detection
    ```

2. **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Download or place your pre-trained U-Net model weights** in the project directory. Ensure the path to the weights file is correct in `main.py`.

2. **Run the Streamlit app**:
    ```bash
    streamlit run main.py
    ```

3. **Upload an MRI image**:
   - Click on "Browse files" to upload an MRI image.
   - Click the "Predict" button to start the segmentation process.

4. **View the results**:
   - The uploaded MRI image and the segmented tumor image will be displayed side by side.

## File Structure
brain-tumor-detection/  
│  
├── app.py # Main Streamlit app script  
├── unet_model.py # U-Net model definition   
├── requirements.txt # Required dependencie  
├── README.md # Project documentation  
└── unet_weights.h5 # Pre-trained U-Net model weights (to be downloaded/placed)  

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The U-Net architecture is inspired by the paper ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/abs/1505.04597) by Olaf Ronneberger, Philipp Fischer, and Thomas Brox.
- Streamlit for providing an easy way to create web apps for machine learning models.
