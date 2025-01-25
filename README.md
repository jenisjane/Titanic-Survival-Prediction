# Titanic-Survival-Prediction
This repository contains my solution to the Kaggle competition Titanic - Machine Learning from Disaster(Link:https://www.kaggle.com/competitions/titanic/overview).  The goal of the project is to predict passenger survival on the Titanic using machine learning techniques. The dataset provides by the project itself.

# Titanic: Machine Learning from Disaster

This repository contains my work on the Kaggle competition [**Titanic - Machine Learning from Disaster**](https://www.kaggle.com/competitions/titanic).

## Project Overview
The objective of this project is to predict which passengers survived the Titanic disaster using machine learning techniques.

---

## Purpose  
This project aims to:  
- Gain insights into the Titanic dataset through thorough exploration and data cleaning, removing irrelevant or inconsistent entries.  
- Leverage **Gradient Boosted Trees (GBT)** to effectively analyze structured data and predict passenger survival.  
- Enhance model performance through feature preprocessing and engineering, ensuring meaningful and well-processed input data.  
- Optimize the model's predictive accuracy by systematically tuning hyperparameters and evaluating results.  


## Workflow
Below is the step-by-step process followed in this project:
### 1. **Import Necessary Libraries**
   - Libraries used:
     - `numpy`: For linear algebra and numerical operations.
     - `pandas`: For data manipulation and processing CSV files.
     - `re`: For regular expression-based text processing.
     - `plotly.express`: For creating interactive visualizations and plotting relationships between features.
     - `tensorflow`: For building, training, and evaluating the machine learning model.
     - `tensorflow_decision_forests`: For using decision forest models within TensorFlow.
     - `random`: For generating random data or operations.
     - `itertools.product`: For generating Cartesian products (useful in hyperparameter tuning).
       
### 2. **Upload and Explore the Dataset**
   - Upload the Titanic dataset (train and test).
   - Display the first few rows of the train dataset to get an overview of its structure and features.
   - Explore statistical summary of the numerical features on the train dataset, showing measures like mean, standard deviation, min, max, and quartiles, which helps in understanding the distribution of the data.

### 3. **Data Cleaning**
   - **Handle Missing Data**: Identify and handle missing values in both the train and test datasets.
   - **Tokenization**:
     - Split the `Name` and `Ticket` columns into smaller pieces (e.g., extract titles like "Mr.", "Mrs.", "Miss").
   - **Remove Irrelevant Features**: Remove columns that do not contribute to the target prediction.

### 4. **Explore Relationships Between Features and Target**
   - **Visualize Feature Relationships**: Plot the relationships between input features and the target variable `Survived`.
     - Use graphs like histograms, box plots, or scatter plots to visualize how each feature influences survival.
   - **Feature Selection**
   - Remove unnecessary columns that are irrelevant or don't contribute meaningfully to the prediction.
   - For example, remove:
     - `PassengerId`: The passenger's unique ID, which does not contain predictive information.
     - `Ticket`: The ticket information, which may not have a meaningful relationship with survival after tokenization.
     - `Survived`: The target variable should not be used as an input feature.
   - The remaining columns are then considered as the input features for the model.

### 5. **Data Preprocessing**
   - **Convert to TensorFlow Dataset**: Convert the cleaned pandas dataset to a TensorFlow dataset for efficient training.
  - **Tokenization**: The "Name" feature is tokenized (split into individual tokens) using `tf.strings.split`, which allows TensorFlow Decision Forests to process text tokens natively.
   - **Categorical Data Handling**: Categorical variables like `Sex`, `Embarked`, and `Pclass` are handled by TensorFlow Decision Forests directly. For models like neural networks, one-hot encoding would be necessary, but TF-DF does not require one-hot encoding for categorical features.
   
### 6. **Model Training**
   - **Define the Model**: Build the machine learning model architecture using TensorFlow/Keras, specifically using the TensorFlow Decision Forests (TF-DF) library for Gradient Boosted Trees.
   - **Hyperparameter Tuning**: 
     - A **random search** approach is used to explore different hyperparameter values for the Gradient Boosted Trees model.
     - The search space includes:
       - `num_trees`: Number of trees in the model.
       - `max_depth`: Maximum depth of each tree.
       - `shrinkage`: Learning rate for the model.
       - `subsample`: Fraction of samples used for training each tree.
       - `min_examples`: Minimum number of samples needed to split a node.
     - The model is trained and evaluated for 20 trials with different random hyperparameter combinations.
     - The best hyperparameters are selected based on the highest accuracy.
   - **Training with Best Hyperparameters**:
     - After identifying the best hyperparameters, the final model is defined and trained with these parameters. 

### 7. **Model Prediction**
   - **Train the Model**: Fit the model on the training data using optimal hyperparameters.
   - **Make Predictions**: Generate predictions on the test dataset.
   - **Threshold Adjustment**: If necessary, adjust the classification threshold for binary predictions.

### 8. **Save and Submit Predictions**
   - **Prepare Submission File**: Format the predictions into the required Kaggle submission format (with `PassengerId` and `Survived`).
   - **Export to CSV**: Save the predictions into a CSV file for submission to Kaggle.
     

## How to Run the Project

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/titanic-machine-learning.git
   cd titanic-machine-learning

---

## Dataset Dictionary
| **Feature**          | **Description**                              |
|-----------------------|----------------------------------------------|
| `PassengerId`         | Unique identifier for each passenger         |
| `Survived`            | Survival indicator (1 = survived, 0 = not)  |
| `Pclass`              | Ticket class (1st, 2nd, 3rd)                |
| `Name`                | Passenger name                              |
| `Sex`                 | Gender                                      |
| `Age`                 | Age of the passenger                       |
| `SibSp`               | Number of siblings/spouses aboard           |
| `Parch`               | Number of parents/children aboard           |
| `Fare`                | Ticket price                                |
| `Embarked`            | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |


---

## Results
- **Final Training Accuracy:** 84.78%  
- **Final Training Loss:** 0.8888  

The Gradient Boosted Trees model demonstrated strong performance, achieving an accuracy of 84.78% on the training data. After fine-tuning the hyperparameters, the model showed a significant improvement in predictive performance. 

---

## Conclusion
This project demonstrates the effectiveness of Gradient Boosted Trees in tackling binary classification problems on structured datasets. The importance of feature engineering and careful tuning of hyperparameters was evident in achieving better results.

---

## Resources
- [Kaggle Titanic Dataset](https://www.kaggle.com/competitions/titanic/overview)
- [Gradient Boosted Trees Documentation (TensorFlow)](https://www.tensorflow.org/decision_forests)

---

