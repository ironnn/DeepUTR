# DeepUTR

DeepUTR is a model designed to predict the translation ability (ribosome profiling reads) of human 5' UTRs. This model aims to analyze the 5' UTR sequences to predict their regulatory capacity on translation.


## Train and inference

- **Model Training**: Train the model using `train.py`.
- **Inference and Prediction**: Perform inference and TE prediction using `model_inference.py`.

## Installation

Follow the instructions below to set up the environment and install the necessary dependencies.

### Step 1: Clone the Repository

Clone this repository to your local machine and navigate to the project directory:

git clone https://github.com/ironnn/DeepUTR.git

cd DeepUTR

### Step 2: Set Up Conda Environment

The environment dependencies are listed in the environment.yml file. To create a Conda environment with the necessary dependencies, use the following command:

conda env create -f environment.yml


### Step 3: Activate the Conda Environment
conda activate deeputr

