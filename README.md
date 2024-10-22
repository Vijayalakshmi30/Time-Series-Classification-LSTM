# Time Series Classification using LSTM

## Project Overview

The primary goal of this project for the CSC 578 Neural Networks and Deep Learning course is to predict the types of astronomical objects based on time-series data collected from the Large Synoptic Survey Telescope (LSST) using machine learning techniques. The project involves data preprocessing, experimentation with different machine learning models, and performance optimization through techniques such as normalization and class balancing. Several models and techniques, such as Bidirectional LSTM and data preprocessing methods like Min-Max Normalization and downsampling were employed, focusing on improving classification accuracy and model performance.

## Dataset
The [LSST data](https://www.kaggle.com/competitions/csc-578-final-project-fall-2023/overview) is a simulated astronomical time-series data, which was created in preparation for the observations to be gathered by the Large Synoptic Survey Telescope (LSST) scheduled to launch in 2019.  LSST will revolutionize our understanding of the changing sky, discovering and measuring millions of time-varying objects.

"These simulated time series, or light curves are measurements of an object's brightness as a function of time - by measuring the photon flux in six different astronomical filters (commonly referred to as passbands). These passbands include ultra-violet, optical and infrared regions of the light spectrum. There are many different types of astronomical objects (that are driven by different physical processes) that we separate into astronomical classes. This data represents a snap shot of the data available and is created from the train set published in the aforementioned competition. 36 dimension was chosen as it represents a value at which most instances would not be truncated. "

Each instance has 36 features per time step, over a sequence of 6 timesteps, thus consists of 216 values -- as a time-rolled-out instance. 
Instances are labeled with 11 target classes -- ['c-15', 'c-16', 'c-42', 'c-52', 'c-62', 'c-65', 'c-67', 'c-88', 'c-90', 'c-92', 'c-95'].
Dataset consists of a training set containing 3356 instances, and a (held-out) test set containing 1439 instances.

## Features

- **Bidirectional LSTM Model**: Utilized for learning from sequential data with memory cells to capture long-term dependencies.
- **Data Preprocessing**: Techniques like Min-Max Normalization and downsampling were applied for improving model performance.
- **Model Evaluation**: Models were evaluated based on classification accuracy, and various approaches were attempted to optimize results.

## Getting Started

### Prerequisites

- Python 3.8 or above
- Libraries: TensorFlow, Keras, Pandas, NumPy, Matplotlib

### Results

The best-performing model was a Bidirectional LSTM with 32 memory cells, achieving notable classification accuracy. Various data processing techniques like normalization contributed to the improvement in model performance.

### Future Work

- Experiment with additional deep learning models, such as GRU and Transformer architectures.
- Implement advanced hyperparameter tuning techniques.
- Apply transfer learning for potential improvements in prediction accuracy.
