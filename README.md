# Face Recognition Data Cleaning with Explainable AI (XAI)

## Project Summary

This project focuses on enhancing face recognition systems by implementing data cleaning techniques and applying Explainable AI (XAI) methods. The main objectives include:

1. **Data Cleaning**: The project implements techniques to identify and remove anomalies and outliers from the face recognition dataset, specifically the Labeled Faces in the Wild (LFW) dataset. This step aims to improve the quality of the training data, leading to better model performance.

2. **Face Recognition Model**: A face recognition model is developed using PyTorch. The model is trained on the cleaned dataset to ensure accurate identification of faces.

3. **Explainable AI (XAI)**: To provide insights into the model's predictions and the effectiveness of data cleaning techniques, various XAI methods are utilized, including:
   - **SHAP**: SHapley Additive exPlanations to interpret the contributions of features to the model's predictions.
   - **LIME**: Local Interpretable Model-agnostic Explanations to explain individual predictions locally.
   - **S-RISE**: A method that uses perturbations to explain model predictions.

By combining these elements, the project aims to create a robust and interpretable face recognition system that is resilient to data quality issues.
