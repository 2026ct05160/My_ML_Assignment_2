# Adult Income Classification taken for assignment-2

### Execution Instructions

#### Google Colab / Jupyter Notebook

1. Upload `adult.csv` to local drive or Google drive(`MyDrive/ML_Assignment_2/adult.csv`).
2. Open `2026ct05160_ML_Assignment2.ipynb` in Google Colab.
3. Run the first cell and authorize Google Drive access.
4. Update `CSV_PATH` if the dataset is stored elsewhere.
5. Select **Runtime > Run all**, or execute the cells in order.

### Dataset source
- **Dataset:** `adult.csv`
- **Source:** UCI Machine Learning Repository - [Adult Dataset](https://archive.ics.uci.edu/dataset/2/adult) - Loaded into My google drive

### Preprocessing

| Stage                 | Implementation                                                                                        |
|-----------------------|-------------------------------------------------------------------------------------------------------|
| Cleaning              | Trim column names and string values; normalize `?`, blank, and `nan` markers.                         |
| Missing values        | Median imputation for numerical data and most-frequent imputation for categorical data.               |
| Duplicates            | Identify and remove exact duplicate records before splitting.                                         |
| Target                | Normalize labels and encode `<=50K` as `0` and `>50K` as `1`.                                         |
| Categorical encoding  | `OneHotEncoder(handle_unknown="ignore")`.                                                             |
| Feature scaling       | `StandardScaler` for numerical predictors.                                                            |
| Data split            | Stratified 80:20 train-test split with `random_state=42`.                                             |
| Leakage prevention    | Fit imputation, encoding, and scaling only on training data through `Pipeline` and `ColumnTransformer`. |


## a. Problem Statement

The objective is to build a binary machine learning classification system using the Adult Census Income dataset. The system predicts whether an individual's annual income is `<=50K` or `>50K` using demographic and employment-related attributes.

The assignment requires the same preprocessed dataset and 80:20 train-test split to be used with five models: Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN), Gaussian Naive Bayes, and Random Forest. Each model is evaluated using Accuracy, AUC, Precision, Recall, F1 score, and Matthews Correlation Coefficient (MCC).

## b. Dataset Description

- **Dataset:** `adult.csv`
- **Source:** UCI Machine Learning Repository - [Adult Dataset](https://archive.ics.uci.edu/dataset/2/adult) - Loaded into My google drive
- **DOI:** `10.24432/C5XW20`
- **Origin:** Extracted by Barry Becker from the 1994 Census database.
- **Records:** 48,842
- **Predictors:** 14
- **Target:** `class`
- **Target classes:** `<=50K` and `>50K`
- **Feature types:** Numerical/integer and categorical
- **Class distribution before duplicate removal:** 37,155 `<=50K` and 11,687 `>50K`

### Why This Dataset Was Selected

The Adult dataset is suitable for binary classification and contains realistic data-science challenges: mixed numerical and categorical features, missing values, duplicate records, and an imbalanced target. It therefore supports a complete preprocessing, modeling, and evaluation workflow.


## c. Github Repository link
    https://github.com/2026ct05160/My_ML_Assignment_2

## d. Models Used

### Model Configurations

| ML Model Name             | Configuration                                                                                                 |
|-----------------------    |---------------------------------------------------------------------------------------------------------------| 
| Logistic Regression       | `max_iter=2000`, `class_weight="balanced"`, `random_state=42`                                                 |
| Decision Tree             | `max_depth=12`, `min_samples_leaf=5`, `class_weight="balanced"`, `random_state=42`                            |
| KNN                       | `n_neighbors=7`, `weights="distance"`, `n_jobs=-1` |
| Naive Bayes               | Default `GaussianNB` configuration |
| Random Forest (Ensemble)  | `n_estimators=200`, `min_samples_leaf=2`, `class_weight="balanced_subsample"`, `random_state=42`, `n_jobs=-1` |

### Comparison Table

| ML Model Name             | Accuracy   | AUC        | Precision  | Recall     | F1         | MCC        |
|---------------------------|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|
| Logistic Regression       | 0.8071     | **0.9034** | 0.5664     | 0.8337     | **0.6745** | **0.5650** |
| Decision Tree             | 0.7959     | 0.8810     | 0.5493     | 0.8290     | 0.6607     | 0.5459	     |
| KNN                       | **0.8356** | 0.8744     | **0.6713** | 0.6155     | 0.6422     | 0.5366     |
| Naive Bayes               | 0.6057     | 0.8243     | 0.3715     | **0.9314** | 0.5311     | 0.3787     |
| Random Forest (Ensemble)  | 0.7849     | 0.9002     | 0.5311	   | 0.8774     | 0.6617     | 0.5527     |

### Graph Representation of All Metrics
## Target class distibution
<img width="1120" height="830" alt="class_distribution" src="https://github.com/user-attachments/assets/11a3b711-6b4d-4b29-a796-2dd1d1f05f45" />

## Confusion Matrix
<img width="739" height="686" alt="confusion_matrix_logistic_regression" src="https://github.com/user-attachments/assets/ecaca4af-bc49-44f1-8690-0941c24cabfa" />
<img width="719" height="686" alt="confusion_matrix_decision_tree" src="https://github.com/user-attachments/assets/cd246475-0455-4dd2-8f5d-36c6c3aa9e71" />
<img width="749" height="686" alt="confusion_matrix_knearest_neighbors" src="https://github.com/user-attachments/assets/69ca81fd-cfec-4ee0-adde-8a8f0cd9cf21" />
<img width="762" height="686" alt="confusion_matrix_gaussian_naive_bayes" src="https://github.com/user-attachments/assets/f8bfd831-b615-4705-a7ea-ed3341b14e01" />
<img width="719" height="686" alt="confusion_matrix_random_forest" src="https://github.com/user-attachments/assets/4f13af1f-4055-4f5c-a76c-91335b713bc8" />

## Model Comparison
<img width="2308" height="1050" alt="model_comparison" src="https://github.com/user-attachments/assets/c130de93-024d-4d6f-8919-e388a9d8f1c9" />

## ROC Curve Comparison
<img width="1588" height="1226" alt="roc_curve_comparison" src="https://github.com/user-attachments/assets/4296abeb-a3be-4f60-ab8d-03312e837565" />

### Model Performance Observations

| ML Model Name             | Observation about Model Performance |
|---------------------------|-------------------------------------|
| Logistic Regression       | Shows the best overall balanced performance. It has the highest AUC (0.9034), F1-score (0.6745), and MCC (0.5650), along with strong recall (0.8337). This indicates good class separation and a good balance between precision and recall. |
| Decision Tree             | Provides reasonably balanced performance with 79.59% accuracy and 82.90% recall. However, its AUC (0.8810), F1 (0.6607), and MCC (0.5459) are slightly below Logistic Regression, indicating somewhat weaker overall discrimination. |
| KNN                       | Achieves the highest accuracy (83.56%) and precision (0.6713) among all models. However, recall is relatively low at 0.6155, meaning that although its positive predictions are more reliable, it misses more actual positive cases.|
| Naive Bayes               | Achieves the highest recall (0.9314), so it identifies most actual positive cases. However, its low precision (0.3715) and accuracy (60.57%) indicate many false-positive predictions. It is suitable when missing a positive case is more costly than generating false alarms. |
| Random Forest (Ensemble)  | Shows very strong recall (0.8774) and a high AUC (0.9002), indicating good discrimination. Its F1-score (0.6617) and MCC (0.5527) are also competitive, but its precision (0.5311) and accuracy (78.49%) are lower than Logistic Regression and KNN.|

### Overall Winner for the Dataset

**Logistic Regression is the overall winner.**

Although KNN has the highest accuracy, the data is imbalanced, so accuracy alone is not sufficient. Logistic Regression achieves highest AUC, F1-score, and MCC, along with strong recall. This indicates good class separation and a good balance between precision and recall.

### BITS Virtual lab screenshots
<img width="1405" height="775" alt="Screenshot1" src="https://github.com/user-attachments/assets/1eb26797-e564-476d-b38f-abe5fb4ffa06" />
<img width="1366" height="768" alt="Screenshot2" src="https://github.com/user-attachments/assets/790822ef-3bf7-44d4-92fc-d0264108f242" />
<img width="1366" height="766" alt="Screenshot3" src="https://github.com/user-attachments/assets/495e6253-51f4-48a5-9e03-413e71242296" />
<img width="1366" height="768" alt="Screenshot4" src="https://github.com/user-attachments/assets/5c16f518-b888-4ae9-8f86-ee430aa83ecc" />
