# PowerPrediction

## Overview

PowerPrediction is a project aimed at predicting power consumption using various machine learning techniques. The repository includes Jupyter Notebooks for data analysis and model training, as well as HTML, Python, and CSS files for the web interface.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- Predictive models for power consumption
- Data preprocessing and analysis
- Interactive web interface for visualization
- Well-documented code and notebooks

## Installation

To get started with the PowerPrediction project, follow these steps:

1. **Clone the repository:**

    ```sh
    git clone https://github.com/Manvadariya/PowerPrediction.git
    cd PowerPrediction
    ```

2. **Create and activate a virtual environment:**

    ```sh
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

4. **Install Jupyter Notebook:**

    ```sh
    pip install notebook
    ```

## Usage

To use the PowerPrediction project, follow these steps:

1. **Run Jupyter Notebook:**

    ```sh
    jupyter notebook
    ```

2. **Open and execute the notebooks:**

    Navigate to the `notebooks` directory and open the Jupyter Notebooks to explore data analysis and model training steps.

3. **Run the web application:**

    ```sh
    python app.py
    ```

    This will launch the web interface where you can visualize predictions and interact with the model.

## Project Structure

The repository is structured as follows:

```plaintext
PowerPrediction/
├── data/
│   ├── raw/                # Raw data files
│   └── processed/          # Processed data files
├── notebooks/              # Jupyter Notebooks for analysis and modeling
├── src/                    # Source code for the project
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── visualization.py
├── static/                 # Static files for the web application
│   ├── css/
│   └── js/
├── templates/              # HTML templates for the web application
├── app.py                  # Main application script
├── requirements.txt        # Python dependencies
└── README.md               # Project README file
```

## Contributors

- [Manvadariya](https://github.com/Manvadariya)
- [ManilModi](https://github.com/ManilModi)
- [ZeelJavia](http://github.com/ZeelJavia)
