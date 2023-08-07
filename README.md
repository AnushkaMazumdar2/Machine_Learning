# Machine_Learning

# Lab 1:
# Tasks:
1. Data Preprocessing
2. Handling Missing Values and Treating Outliers'
3. Exploratory Data Analysis
4. Data Scaling
5. Dimensionality Reduction
6. Feature Selection
7. Handling Class Imbalance

# CODE DESCRIPTION

# STEP 1: IMPORTING NECESSARY LIBRARIES
Firstly I imported all the necessary libraries that will be used for my analysis.
1. pandas: It has functions for analysing, cleaning and manipulating data
2. numpy: It is used to work with arrays 
3. seaborn: It is a python data visualization tool for making statistical graphs
4. matplotlib: It is used to create static and interactive visualisations in python
5. sklearn: It is used to implement machine learning models and statistical modelling
7. StandardScaler: It scales the data using min-max normalisation/z-score normalization
8. Pipeline: It is used to concatenate severl steps that can be cross-validated together while setting different parameters
9. PCA: It is a linear dimensionality reduction technique
10. train-test-split: It is used to split the dataset on the basis of the target variable
11. GridSearchCV: It is used to find the optimal parameter values from a given set of parameters across a grid
12. sklearn.metrics: These are metrics used to judge a model's performance. It includes precision, accuracy, recall, etc.
14. LabelEncoder: It is used to convert categorical data to numerical format
15. DecisionTreeClassifier: This is a supervised learning method used for classification 

# STEP 2: IMPORTING THE DATASET AND UNDERSTANDING THE DATA
Firstly I imported the csv file using pandas. Then to initially understand the features and study the target variable, I used some methods:
1. .head() :This is used to print the first 5 rows of the dataset. Here we can see that there are many features, including the target variable, "Exited"
2. .info() : This is used to get an overview of our data and the data types of the features used. In our data, there are 10,000 rows and 14 columns. "Object" data types signifies string values, "int" signifies integer values and "float" signifies decimal values
3. .describle() : This gives the statistical summary of our data, how the features are distributed. For e.g. credit score has a mean of 249.5, and its minimum value is 0 and maximum value is 449, its 1st quartile is 183, 2nd quartile is 251, 3rd quartile is 331

# STEP 3: HANDLING MISSING DATA
I used the function .isnull() to check if all the features in the dataset has any missing values or not, .sum() calculates the number of missing values each attribute might have and .toframe() converts the given output to a table for easy understanding. We can see by the results that there are no missing values in our dataset.
I also checked for duplicate values across all the columns. Thankfully there were none. Then upon manual inspection, I found three columns: "RowNumber","Surname","CustomerId" to be redundant in my analysis so I decided to drop those columns using .drop()

# STEP 4: HANDLING OUTLIERS/NOISY DATA
Outlier Detection:
One of the best ways to detect outliers in our data is by building boxplots. I created a for loop for creating boxplots for all the subsequent attributes using the .plot() function and the kind as "box". Upon close inspection, we find that there are two features with outliers: "CreditScore" and "Age".
Outlier Treatment:
The outliers were dealth with one of the popular methods known as IQR(Inter Quartile Range). First the 1st quartile and 3rd quartile of the variable is calculated, then on that basis the inter quartile range is computed. Then the lower limit and upper limit of the variable is calculated. The "width" of IQR for outlier detection is 1.5 because a smaller value would mistake the outliers as data points and a larger value would mistake some of the data points as outliers. Now, the values that are above the upper limit is then replaced with the upper limit value and same happens with the lower limit value. When we again check the bosplot, we see that the outliers have been treated


# RITHUL'S WORK IS DONE! PHEW!

# STEP 5: EXPLORATORY DATA ANALYSIS
1. Firstly I drew a categorical plot using sns.catplot() for my target variable and we can see that there is a huge imbalance between the people who left the bank and who didn't. This type of imbalance can introduce bias in our data and decrease the accuracy of our model. We will deal with class imbalance later in this notebook
2. BINNING:
   a) I created a new feature called "TotalProducts" using "NumOfProducts". Through putting conditions, I categorised the variable into three categories for easy analysis
   b) I also categorised the "Age" feature into two categories:"Adult", "Senior Adult"
3. DISCRETE PLOTS:
   I created a function called countplot() to find the relationship between variables
   a) "Customer Churned By Gender": The churned probabillity is more for Feamle Customers compared to male customers, which means more female customers are deactivating their bank accounts compared to male customers.
   b) "Customer Churned By Geography": Despite the huge total customers difference between France & Germany the churned rate for France and Germany customers are same. The Churn rate is almost double in Germany as compared to Spain.
   c) "Customer Churned by HasCrCard" : This analysis is done on the basis of a person having a credit card or not. There are more people having credit card but leaving the firm
   d) "Customer Churned by IsActiveMember" : More active members are leaving the firm
   e) "Customer Churned by Tenure" : There is overall not much difference between the tenure and customer churn. There are customers who have bank account in this bank for 10 years and still leaving the bank. No further analysis can be done on this
   f) "Customer Churned by TotalProducts" : The majority of the customers only have 1 or 2 bank products. Customers with less than 2 products have the least churn rate.
4. CONTINUOUS PLOTS:
   I created a function called continous plots which are essentially boxplots to check how the median value for the continous variables are differing
   a) "Distribution of CreditScore by Churn Status" : The Median Credit score for both churned and not churned is almost equal
   b) "Distribution of Age by Churn Status" : Older aged people are deactivating their bank account more
   c) "Distribution of Balance by Churn Status" : Customers with low account balance are more likely to deactivate their bank accounts
   d) "Distribution of Estimated Salary by Churn Status" : The median salary for both churned and not churned custmers is almost same. So no inference can be drawn.
5. Heatmap:
   The heatmap is used to check which variables are strongly or weakly correlated with one another. There are not many strong correlation betweem these variables

# TAMIRRA'S WORK IS ALSO DONE! YAY!

# STEP 6: DATA SCALING
All the categorical data has been converted to numerical data. After that using normalization techniques, the data has been scaled so as to remove any bias from the data

# STEP 7: DIMENSIONALITY REDUCTION
First the function PCA() is called and then the pipeline function is used to first scale the data and then apply principal component analysis on it. Then using fit_transform, the dataset is transformed into a form more suitable for the model.
Using plt.scatter(), the scatterplot of the two classed is visualised after PCA
I split the dataset into training and testing set

# STEP 8: FEATURE SELECTION
Using Chi-Square test, we will find the features that affect our target variable the most. We import chi2. Then we perform chi-square test on all the categorical variables across the dataset against the output variable, "Exited". Then we plot the chi-square values and find that "Total_Products","IsActveMember" and "Gender" are the top three features affecting the dataset.

# STEP 9: CLASS IMBALANCE HANDLING
Using SMOTE(Synthetic Minority Over-sampling Technique), I reduced the class imbalance and successfully resampled the data

# STEP 10: MODEL TRAINING AND PREDICTION
I used Decision Tree algorithm for my dataset. The param_grid variables store the best parameters of my model. GridSearchCV find the optimal parameters for my model. Then i fit the model on my training data and predict the outcome using my test data. The accuracy of my model was 79.08%. Upon calculating the best features, the same result as my chi-square scores came, with "Total_Products","Age", and "IsActiveMember" came as top features.
I also drew a confusion matrix for my data. The True Negative Value is: 1686
The True Positive Value is: 293
The False Negative Value is: 324
The False Positive Value is: 197

# TANILA'S WORK IS ALSO DONE!!

# THANK YOU MANAGER FOR SUCH A WONDERFUL OPPORTUNITY TO BRUSH UP ON MY DATA PREPROCESSING SKILLS!


