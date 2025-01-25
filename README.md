# PRODIGY_ML_02 - Customer Segmentation using K-Means Clustering
# Overview
This project implements a K-Means Clustering algorithm to group customers based on their purchase history. The objective is to segment customers into different groups, which can then be used for targeted marketing or personalized recommendations. The model is trained using a retail customer dataset, and customers are grouped based on features such as annual income and spending score.

# Features
The model uses the following features from the dataset to group customers:

Annual Income: The annual income of the customer.
Spending Score: A score assigned to each customer based on their spending behavior.
Dataset
The dataset used in this project is provided by Kaggle and contains various customer-related features, including:

CustomerID: A unique identifier for each customer.
Gender: The gender of the customer.
Age: The age of the customer.
Annual Income (k$): The annual income of the customer in thousands of dollars.
Spending Score (1-100): A score assigned to each customer based on their purchasing behavior.
You can download the dataset from Kaggle: Customer Segmentation Dataset.

# How to Run the Project
1. Install Dependencies
Make sure you have Python installed, then install the required libraries using pip:

bash
Copy
pip install pandas scikit-learn matplotlib seaborn
2. Download the Dataset
Download the dataset from Kaggle and place the CSV file Customer Segmentation.csv in the project directory.

3. Run the Model
Once the dataset is available, you can run the model using the following Python code:

python
Copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('Customer Segmentation.csv')

# Show the first few rows of the dataset to understand its structure
print(data.head())

# Select relevant features (Annual Income and Spending Score)
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering (let's assume 5 clusters for this example)
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)

# Add the cluster labels to the original data
data['Cluster'] = kmeans.labels_

# Visualize the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data['Annual Income (k$)'], y=data['Spending Score (1-100)'], hue=data['Cluster'], palette='viridis', s=100)
plt.title('Customer Segmentation based on Annual Income and Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()

# Display cluster centers
print("Cluster Centers:")
print(kmeans.cluster_centers_)

# Optionally, analyze the distribution of customers within each cluster
cluster_summary = data.groupby('Cluster').agg({'Annual Income (k$)': ['mean', 'std'], 'Spending Score (1-100)': ['mean', 'std']})
print(cluster_summary)
4. Results
After running the model, the following results will be displayed:

Customer Segmentation Visualization: A scatter plot of customers grouped by their annual income and spending score, with different clusters shown in different colors.
Cluster Centers: The centroids of each cluster in the feature space (Annual Income, Spending Score).
Cluster Summary: A summary of the mean and standard deviation of the features for each cluster.
5. Visualizing the Results
The customer segmentation plot will display clusters of customers based on their annual income and spending score. You can see the distribution of customers in various income and spending score categories.

# Screenshot of the output:
![Screenshot 2024-09-07 185851](https://github.com/user-attachments/assets/4d699966-c9eb-4e06-8ff1-4c3b9211f457)

![Screenshot 2024-09-07 185906](https://github.com/user-attachments/assets/f09c3c82-4ccf-4e8e-9f23-7f586aff3bc1)


# Evaluation
The K-Means algorithm does not require labeled data for training, making it an unsupervised learning technique. However, we can evaluate the quality of the clustering using the following metrics:

Inertia: Measures how well the data points fit within their assigned clusters. A lower inertia indicates better clustering.
Cluster Centers: The centroids of the clusters, showing the average annual income and spending score for customers in each cluster.
Cluster Distribution: The number of customers in each cluster, which can help understand the segmentation behavior.
python
Copy
# Inertia (Sum of squared distances of samples to their closest cluster center)
print(f"Inertia: {kmeans.inertia_}")
6. Predicting New Customer Segments
To predict which segment a new customer belongs to, you can use the following function:

python
Copy
def predict_segment(income, spending_score):
    new_data = np.array([[income, spending_score]])
    new_data_scaled = scaler.transform(new_data)
    cluster = kmeans.predict(new_data_scaled)
    return cluster[0]

# Example: Predict the segment for a new customer
new_income = 80
new_spending_score = 50
predicted_segment = predict_segment(new_income, new_spending_score)
print(f"The new customer belongs to cluster: {predicted_segment}")
7. Folder Structure
After following the steps, your project directory should look like this:

# Conclusion
This project demonstrates how to use K-Means Clustering to group customers based on their purchasing behavior. By clustering customers, we can create targeted marketing strategies or personalize product recommendations. The model could be further improved by incorporating additional features, such as the customer's purchase history, demographics, or browsing behavior.
