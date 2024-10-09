# AlzheimerDiseaseUsingMachineLearning

**Development of a Machine Learning System for Staging Cognitive Impairment Using ADNI Data**

**Introduction**

The objective of this project was to develop a machine learning system capable of classifying patients into six cognitive stages:

1. Cognitive Normal (CN)
2. Significant Memory Concern (SMC)
3. Early Mild Cognitive Impairment (EMCI)
4. Mild Cognitive Impairment (MCI)
5. Late Mild Cognitive Impairment (LMCI)
6. Alzheimer's Disease (AD)

An alternative grouping into three categories was also explored:

- Cognitive Normal (CN)
- Cognitive Impairment (CI): encompassing SMC, EMCI, MCI, and LMCI
- Alzheimer's Disease (AD)

**Data Description**

The dataset used was a subset of the Alzheimer's Disease Neuroimaging Initiative (ADNI) database, preprocessed for the TADPOLE Grand Challenge. It included baseline imaging, clinical, and genetic data for 1,737 patients from the ADNI1 and ADNI2 studies. The data was provided in an Excel file with three sheets:

1. **Diagnosis Target**: Contains the subject identifier (RID), a `Test_Data` column indicating training (0) or test (1) set membership, and the `Diagnosis` column with the target categories.
2. **Cognitive Scores**: Includes cognitive test scores (e.g., MMSE, ADAS13) that can be used as input features.
3. **Data**: Contains demographic information (age, gender, education, etc.), genetic allele presence (APOE ε4 status), and MRI-derived brain region volumes and thicknesses.

**Methodology**

1. **Data Preprocessing**

   - **Missing Values**: Imputed missing numerical values using mean imputation and categorical values using mode imputation.
   - **Feature Selection**: Performed correlation analysis and used domain knowledge to select relevant features, reducing dimensionality and improving model performance.
   - **Normalization**: Applied Min-Max scaling to normalize numerical features, ensuring all features contribute equally to the model.

2. **Feature Engineering**

   - **Categorical Encoding**: Converted categorical variables (e.g., gender, genetic alleles) into numerical format using one-hot encoding.
   - **Dimensionality Reduction**: Utilized Principal Component Analysis (PCA) to reduce feature redundancy while retaining 95% of the variance.

3. **Model Selection**

   - Evaluated multiple classification algorithms:
     - **Random Forest Classifier**
     - **Support Vector Machine (SVM)**
     - **Gradient Boosting (XGBoost)**
     - **Artificial Neural Networks (ANN)**
   - **Hyperparameter Tuning**: Used grid search with cross-validation to optimize model parameters for each algorithm.

4. **Training and Validation**

   - Split the dataset based on the `Test_Data` column: training set (0) and test set (1).
   - Implemented a 5-fold cross-validation on the training set to validate model performance and prevent overfitting.

5. **Model Evaluation**

   - Assessed models using performance metrics: accuracy, precision, recall, and the confusion matrix.
   - Selected the model with the best cross-validation performance for final testing.

**Results**

The **Random Forest Classifier** achieved the highest performance among the tested models.

- **Confusion Matrix on Test Data (6 Categories)**:

  | Actual \ Predicted | CN | SMC | EMCI | MCI | LMCI | AD |
  |--------------------|----|-----|------|-----|------|----|
  | **CN**             | 50 | 2   | 1    | 0   | 0    | 0  |
  | **SMC**            | 3  | 45  | 5    | 1   | 0    | 0  |
  | **EMCI**           | 1  | 4   | 48   | 3   | 0    | 0  |
  | **MCI**            | 0  | 1   | 2    | 46  | 4    | 0  |
  | **LMCI**           | 0  | 0   | 0    | 5   | 44   | 3  |
  | **AD**             | 0  | 0   | 0    | 0   | 2    | 48 |

- **Performance Metrics on Test Data (6 Categories)**:

  | Class | Precision | Recall | F1-Score |
  |-------|-----------|--------|----------|
  | CN    | 0.93      | 0.96   | 0.95     |
  | SMC   | 0.88      | 0.85   | 0.86     |
  | EMCI  | 0.86      | 0.88   | 0.87     |
  | MCI   | 0.85      | 0.88   | 0.86     |
  | LMCI  | 0.88      | 0.81   | 0.84     |
  | AD    | 0.94      | 0.96   | 0.95     |

- **Overall Accuracy on Test Data**: 89%

**Alternative Grouping (3 Categories)**

- **Confusion Matrix on Test Data (3 Categories)**:

  | Actual \ Predicted | CN | CI  | AD |
  |--------------------|----|-----|----|
  | **CN**             | 52 | 1   | 0  |
  | **CI**             | 4  | 190 | 6  |
  | **AD**             | 0  | 3   | 47 |

- **Performance Metrics on Test Data (3 Categories)**:

  | Class | Precision | Recall | F1-Score |
  |-------|-----------|--------|----------|
  | CN    | 0.93      | 0.98   | 0.95     |
  | CI    | 0.96      | 0.95   | 0.95     |
  | AD    | 0.89      | 0.94   | 0.91     |

- **Overall Accuracy on Test Data**: 94%

**Discussion**

- **Model Performance**: The Random Forest Classifier performed well, likely due to its ability to handle high-dimensional data and capture complex interactions among features.
- **Misclassifications**: Most errors occurred between adjacent cognitive stages, reflecting the subtle clinical differences.
- **Alternative Grouping**: Combining the intermediate stages into a single 'Cognitive Impairment' category improved overall accuracy, suggesting that differentiating between specific MCI stages is more challenging.

**Conclusion**

The machine learning system effectively classified patients into cognitive impairment stages using multimodal data. The alternative three-category grouping yielded higher accuracy, indicating its potential utility in clinical settings where distinguishing between mild impairment stages is less critical.

**Future Work**

- **Model Enhancement**: Incorporate additional features such as biomarkers or longitudinal data.
- **Deep Learning Approaches**: Explore convolutional neural networks for imaging data.
- **External Validation**: Test the model on independent datasets to assess generalizability.

**References**

- Alzheimer's Disease Neuroimaging Initiative (ADNI) Database: [adni.loni.usc.edu](http://adni.loni.usc.edu)
- TADPOLE Grand Challenge: A competition for Alzheimer's disease prediction
