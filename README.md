# Cardiac Segmentation ACDC

Deep learning-based cardiac segmentation using PyTorch, MONAI, and U-Net models.

## About
This project leverages deep learning models for cardiac segmentation, particularly for the ACDC dataset. The goal is to segment the heart's anatomical structures to aid in medical analysis.

### Features:
- **2D and 3D U-Net models** for segmentation.
- **Attention mechanisms** for improved segmentation accuracy.
- **Support for both 2D and 3D datasets** (using MONAI and PyTorch).

## Installation

### Prerequisites:
- Python 3.x
- PyTorch
- MONAI
- other Python dependencies from `requirements.txt`

### Setup:
1. Clone the repository:
    ```bash
    git clone https://github.com/arkanandi/Cardiac_Segmentation_ACDC.git
    cd Cardiac_Segmentation_ACDC
    ```

2. Install the required Python libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the ACDC dataset and place it in the appropriate folder (see the dataset instructions in the repository for details).

4. To train the model, run:
    ```bash
    python train_2d.py
    # or
    python train_3d.py
    ```

5. To make predictions:
    ```bash
    python predict_2d.py
    # or
    python predict_3d.py
    ```

## Usage

Once you have trained the model, you can use it to predict cardiac structures on new datasets.

## Contributing

Feel free to fork the repository and submit pull requests. Issues and suggestions are always welcome.

1. Fork this repository.
2. Clone your fork:
    ```bash
    git clone https://github.com/your-username/Cardiac_Segmentation_ACDC.git
    ```
3. Create a new branch:
    ```bash
    git checkout -b feature-name
    ```
4. Make changes and commit:
    ```bash
    git commit -am 'Add new feature'
    ```
5. Push to your fork:
    ```bash
    git push origin feature-name
    ```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
