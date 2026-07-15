# Feature_Engineering_Sprint_9
Transforming raw data into powerful predictive signals through encoding, scaling, polynomial features, and intelligent feature selection.

## Feature engineering
 is the art and science of transforming raw data into the optimal input format for machine learning algorithms. It's where domain knowledge meets technical skill, and where good practitioners separate themselves from great ones.

## What we'll learn:

* Feature engineering- transforming raw data into powerful predictive signals through encoding, scaling, polynomial features, and intelligent feature selection

* Imbalanced data solutions
handling real-world scenarios where classes are unequal, using sampling techniques, threshold adjustment, and advanced evaluation metrics

*Professional model evaluation

using learning curves, validation curves, and ROC analysis to understand model behavior and make deployment decisions.

## What We’ll be learning:

Data Cleaning and Preparation
Missing value strategies
    When to fill, when to drop, and how different approaches affect model performance
Feature scaling and normalization
    Why some algorithms fail without proper scaling and how to choose the right technique
Categorical Data Mastery
    One-hot encoding
        Converting categories like "Red", "Blue", "Green" into numerical formats algorithms can understand

    Label and ordinal encoding
        Handling categories with natural ordering like "Small", "Medium", "Large"
    The dummy variable trap
        A subtle but critical mistake that can ruin model performance
Advanced Feature Creation
    Polynomial features
        Creating interaction terms that capture complex relationships between variables
    Feature selection
        Identifying which features actually improve predictions and eliminating noise
    Feature importance analysis
        Understanding which data points drive your model's decisions
Professional Workflows

    ML pipelines
        Building automated systems that prevent data leakage and ensure reproducible results

## machine learning algorithms only understand numbers.
Understanding Categorical Data Types

Nominal Categories (No Natural Order)
Characteristics: Categories with no inherent ranking or order

Examples from real estate:

Neighborhood: Downtown, Suburbs, Rural (no clear "best" to "worst" order)
Property_Type: House, Apartment, Condo (different types, not ranked)
Exterior_Color: Red, Blue, White, Gray (purely descriptive)
Key insight: There's no mathematical relationship between categories. Blue isn't "greater than" Red.

Ordinal Categories (Natural Order Exists)
Characteristics: Categories with clear ranking or hierarchy

Examples from real estate:

Condition: Poor < Fair < Good < Excellent (clear quality progression)
Size_Category: Small < Medium < Large (obvious ordering)
School_Rating: F < D < C < B < A (grade-based hierarchy)
Key insight: Order matters, and the algorithm should understand this ranking.
 
 The Solution Preview
Different categorical types require different handling approaches:

For Nominal Categories (no order):

One-Hot Encoding: Create separate yes/no columns for each category
Preserves all information without implying false relationships
For Ordinal Categories (natural order):

Label Encoding: Assign meaningful numbers that respect the natural ordering
Ordinal Encoding: More sophisticated approach with proper spacing

## Systematic Feature Type Assessment

The Two-Stage Assessment Strategy
Professional data scientists check feature types twice during every project:

Stage 1: Initial Dataset Assessment
When: Immediately after loading the dataset
Purpose: Get familiar with your data structure
Questions to answer:

How many features do I have?
Which appear to be categorical vs numerical?
Are there obvious data quality issues?

Stage 2: Post-Preprocessing Validation

When: After handling missing values, duplicates, and manual type conversions
Purpose: Sanity check before model training
Questions to answer:

Did my preprocessing steps work correctly?
Are all features in the expected format?
Are categorical features properly encoded?


Encoding Overview:
 OHE :in regression, the categories have to be encoded in a way that assigns equal importance to all of them.One-Hot Encoding (OHE) for nominal variables — it creates separate yes/no columns without implying any order!
 For tree-based algorithms, cateogories are often strings we should convert them to numbers . Here is when label Encoding comes into play.

  As for ordinal variables, it is better to use Ordinal Encoding, regardless of the algorithm.
  It is used in a tree-based algorithm (some machine learning frameworks support unencoded categorical variables for trees, however, if they don't, Label Encoding is an option).
It is an ordinal variable (Ordinal Encoding should be used for ordinal variables).
It is a high cardinality variable (more advanced encoding techniques might be needed).


=========================================================================
## Classification Metrics: Accuracy for the Decision Tree
Accuracy is a metric that is used for classification tasks. 
Accuracy can be calculated with the accuracy_score() function available in the sklearn.metrics module. It takes two arguments: correct targets and predictions, and returns the calculated accuracy score.

### Sanity Cehck
To assess the accuracy of the model, let's first examine the targets in our dataset. Since we have a binary classification problem, the targets can only be either 0 or 1. We are interested to know how many 0s and 1s are in the dataset that we are working with. To do that, we will use the value_counts() method learned previously.

You know that the count for each unique value in a given column can be calculated using value_counts() in the following manner: data['particular_column'].value_counts(). As an option, we can consider setting True to the normalize= parameter within the value_counts() method to get the output in relative units.

### True Positive
What does a True Positive answer (TP) mean? In this case, the model labeled an observation as 1, and its actual value is also 1. 

If the predicted and actual class values are negative, then the answer is True Negative.

### True Negative 
If the predicted and actual class values are negative, then the answer is True Negative.

In our task, the True Negative answer (TN) is the number of insured people who:

according to the model's prediction did not request a payment, AND
didn't actually apply for insurance compensation.

### False Postivie 
 
 Algorithms are like humans — they make errors. These errors fall into two categories.

The first type of error is a False Positive (FP). It occurs when the model predicted "1", but the actual value of the class is "0".


### False Negative

The second type of error are False Negative answers (FN).

False Negative answers occur when the model predicts "0", but the actual value of the class is "1".

In our task, a False Negative answer is the number of insured people who:

according to the model's prediction did not request a payment, BUT
in fact, did make a claim.

### Confusion Matrix

The confusion matrix is formed as follows:

The shape of the matrix is created
labels predicted by an algorithm (0 and 1) are placed on the horizontal axis ("Predictions");

actual labels (0 and 1) are placed on the vertical axis ("Answers").
   
    -------------------------
        TP      |    FN
   ------------ |------------
       FP       |    TN
  --------------------------
![alt text](image.png)

Now we have a 2 by 2 grid, but it is not a confusion matrix yet. To become a confusion matrix, it needs to be filled with the values of our four possible outcomes. Therefore, our second step is...

Filling in the matrix with the TP, FP, TN and FN values using the following rules:
The correct predictions are placed on the main diagonal (from the upper-left corner):
TN in the upper-left corner
TP in the lower right corner
Incorrect predictions are outside of the main diagonal:
FP in the upper right corner
FN in the lower-left corner

### Recall  # of all the TP how many model successfully find 
 the proportion of positive answers marked positive by the model (TP) to the positive answers marked positive by the model (TP) plus the answers marked negative by the model that are actually positive (FN). 
 
                    Recall = TP/TP+FN

### precision: When I say How often am I right?

Precision measures how many negative answers the model found while searching for positive ones. 
                   
                   percision = TP/TP+FP

**`Similar to recall, we want the precision to be close to one.`**

F1Score:
Recall and precision evaluate the quality of predictions of the positive class from different angles. Recall describes how well a model understands the properties of this class and is able to recognize it. Precision detects whether a model is overdoing the positive class recognition by assigning too many positive labels.

Both recall and precision are important, but sometimes we need a little more. Here is where we can use an aggregating metric, the F1 score, to work with both recall and precision simultaneously. Basically, the F1 score is the harmonic mean of recall and precision.

F1 score =2×Precision × Recall/ Precision + Recall

It's important to understand that when either recall or precision is close to zero, the harmonic mean is close to 0.

## Impbalanced_Classification 
1- **Class Weight Adjustment**
    By default, machine learning algorithms consider all observations in the training set to be equally weighted.
    *`By default, machine learning algorithms consider all observations in the training set to be equally weighted.`* 
     The `logistic regression algorithm` in the `sklearn library` has the `class_weight` argument. By default, it is `None` — i.e., `classes are equivalent`, meaning that:
     class_weight='balanced' as telling the algorithm:

"Minority class examples are more valuable. If you misclassify one of them, treat it as a much bigger mistake than misclassifying a majority-class example."

**How can we increase the number of samples in the minority class?**
 * `upsampling`
 
Upsampling is performed in several steps:
Splitting the training dataset into negative and positive observations
Duplicating the positive observations (the ones that have rare occurrences) several times
Appending duplicated observations to the training dataset
Shuffling the new tr

**When to use it**

* Use it when the classes are imbalanced, especially when class 1 is much   smaller than class 0.
* Use it only on the training data to avoid data leakage.
*  Use this on the training data only, after train_test_split and before model.fit().

# Upsampling function
def upsample(features, target, repeat):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

 # It duplicates the minority class with pd.concat().
    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)
# It shuffles the result with shuffle(..., random_state=12345).
    features_upsampled, target_upsampled = shuffle(
        features_upsampled, target_upsampled, random_state=12345
    )

    return features_upsampled, target_upsampled

# Call the function for the training set only.
features_upsampled, target_upsampled = upsample(X_train, y_train, 10)

print(features_upsampled.shape)
print(target_upsampled.shape)

## Downsampling
How can we reduce the number of samples in the majority class?
![alt text](image-1.png)

`Downsampling is performed in several steps:`

* Split the training dataset into negative and positive observations
* Randomly drop a portion of the negative observations
* Create a new training sample based on the data obtained after the drop
* Shuffle the data. This will make sure that the positive data doesn't follow the negative. Doing that will ease the training job for any machine learning algorithm.

To *discard* some of the data elements, use the `sample()` function available in pandas. It takes the frac parameter (from fraction), which specifies the relative portion of the data that you want to end up with after a drop. Then the `sample()` function randomly drops the data in a DataFrame and returns another DataFrame which has an amount of data you specified in the frac parameter. Here is an example:

print(features_train.shape)

features_sample = features_train.sample(frac=0.1, random_state=12345)
print(features_sample.shape)

Step by step creating downsample
target = data['Col']
features = data.drop('Col', axis=1)
t_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)
## Create def downsample
def downsample(features, target, fraction):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_downsampled = pd.concat(
        [features_zeros.sample(frac=fraction, random_state=12345)]
        + [features_ones]
    )
    target_downsampled = pd.concat(
        [target_zeros.sample(frac=fraction, random_state=12345)]
        + [target_ones]
    )

    features_downsampled, target_downsampled = shuffle(
        features_downsampled, target_downsampled, random_state=12345
    )

    return features_downsampled, target_downsampled


features_downsampled, target_downsampled = downsample(
    features_train, target_train, 0.1
)

model = LogisticRegression(random_state=12345, solver='liblinear')
model.fit(features_downsampled, target_downsampled)
predicted_valid = model.predict(features_valid)
print('F1:', f1_score(target_valid, predicted_valid))

3- Classification Threshold

logistic regression models return a probability that a given observation is positive. So, let's take a closer look at the probabilities that the logistic regression model outputs. 

As with any other probability, it can only have a range from zero to one. 

The probability level at which the negative class ends and the positive class begins is called the threshold. By default, the initial threshold is held at 0.5. If the probability is greater than or equal to this, then the observation is positive; otherwise, it's negative

* Threshold adjustment
In sklearn, the class probability can be calculated with the `predict_proba()` function. It takes features from some observations and returns the probabilities for each observation:
As we change the threshold value, we will see our metrics change as well.

In sklearn, the class probability can be calculated with the predict_proba() function. It takes features from some observations and returns the probabilities for each observation:

probabilities = model.predict_proba(features)

Takeaway:

predict_proba() returns the probability that each observation belongs to each class. In binary classification, the first column is the probability of Class 0 (negative class), and the second column is the probability of Class 1 (positive class). The probabilities in each row always add up to 1 (100%) because every observation must belong to one of the two classes. By default, if the probability of the positive class is 0.5 or higher, the model predicts Class 1; otherwise, it predicts Class 0. This is useful for understanding the model's confidence and for adjusting the decision threshold when needed.

### PR curve
On the graph, the precision values are on the vertical axis, whereas the recall values are along the horizontal axis.
![alt text](image-2.png)
A curve plotted from Precision and Recall values is called a PR curve. The higher the area under the curve, the better the model.

### TPR & FPR 
True Positive Rate, or TPR, is the result of the TP answers divided by all positive answers. Here's the formula, where P is all positive answers:

                TPR=TP/P


The False Positive Rate, or FPR, is the result of the FP answers divided by all negative answers. It is calculated using a similar formula, where N is all negative answers:

FPR = FP/N
We plot the false positive rate values (FPR) along the horizontal axis and true positive rate values (TPR) along the vertical axis. Then we iterate over a range of threshold values and plot a point for each threshold. This generates a curve called the ROC curve (Receiver Operating Characteristic).
The ROC curve model always makes random predictions in a diagonal line going from the lower left to the upper right. The higher the area under the curve, the better the model's quality.
​
![alt text](image-3.png)
To find how much our model differs from the random model, let's calculate the AUC-ROC value (Area Under Curve ROC). This is a new evaluation metric with values in the range from 0 to 1. The AUC-ROC value for a random model is 0.5.
We can plot a ROC curve with the roc_curve() variable from the sklearn.metrics module:
from sklearn.metrics import roc_curve
It takes the target values and the positive class probabilities, goes over different thresholds, and returns three lists: FPR values, TPR values, and the different thresholds it went over.
We have just witnessed a new showdown: TPR vs. FPR. Let's plot the curve.

We plot the false positive rate values (FPR) along the horizontal axis and true positive rate values (TPR) along the vertical axis. Then we iterate over a range of threshold values and plot a point for each threshold. This generates a curve called the ROC curve (Receiver Operating Characteristic).

The ROC curve model always makes random predictions in a diagonal line going from the lower left to the upper right. The higher the area under the curve, the better the model's quality.



To find how much our model differs from the random model, let's calculate the AUC-ROC value (Area Under Curve ROC). This is a new evaluation metric with values in the range from 0 to 1. The AUC-ROC value for a random model is 0.5.

We can plot a ROC curve with the roc_curve() variable from the sklearn.metrics module:

from sklearn.metrics import roc_curve
It takes the target values and the positive class probabilities, goes over different thresholds, and returns three lists: FPR values, TPR values, and the different thresholds it went over.

fpr, tpr, thresholds = roc_curve(target, probabilities)

### Feature Engineering

Introducing Interaction Features :Interaction features capture the combined effect of two or more variables working together. 

##### Polynomial Features: Capturing Non-Linear Relationships

Sometimes the relationship between a feature and your target isn't a straight line. Polynomial features help capture curved relationships by creating powers of existing features.

*Scikit-learn* provides Polynomial Features to automatically generate polynomial and interaction features:

**Import libaray**
`from sklearn.preprocessing import PolynomialFeatures`

** #Create polynomial features up to degree 2**
`poly = PolynomialFeatures(degree=2, include_bias = False)`
`poly_features = poly.fit_transform(numerical_features)`

#### Feature Scaling and Normalization
The Rule:

Scale-sensitive algorithms: Logistic Regression, SVM, Neural Networks, K-means **`# use StandardScaler.`**

Scale-invariant algorithms: Decision Trees, Random Forest, Gradient Boosting  **`# NO Scaler needed.`**

Types of Feature Scaling
StandardScaler (Z-score Normalization)
What it does: Transforms features to have mean=0 and standard deviation=1

Formula: scaled_value = (original_value - mean) / standard_deviation

When to use: Most common choice, works well when data is approximately normally distributed

MinMaxScaler (Min-Max Normalization)
What it does: Transforms features to a fixed range, typically [0, 1]

Formula: scaled_value = (original_value - min) / (max - min)

When to use: When you need features in a specific range, or when you know the bounds of your data

**Scaling in Practice with Scikit-Learn**
The Critical Rule: Fit on Training, Transform on Al

from sklearn.preprocessing import StandardScaler

**CORRECT way - prevents data leakage**

scaler = StandardScaler()
scaler.fit(X_train)  # Learn scaling parameters from training data only

X_train_scaled = scaler.transform(X_train)  # Apply to training data
X_test_scaled = scaler.transform(X_test)    # Apply same scaling to test data

# WRONG way - causes data leakage
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)  # Don't do this!

### Feature Selection:
🎯 Main Idea

After creating new features (polynomial, interaction, one-hot encoding), not every feature helps the model.

Goal: Keep only the most useful features and remove unimportant ones.

❓ Why Feature Selection?

## Too many features can cause:

⏳ Slower model training
🎯 Overfitting (memorizing noise)
📉 Poor prediction on new data
🤔 Harder to understand and explain the model

Sometimes fewer features produce better accuracy.

#### Three Feature Selection Methods
1. Statistical Selection

Uses statistics (correlation, chi-square, etc.)

Keep features that have a strong relationship with the target.
*Calculate correlation between features and target*
correlations = features.corrwith(target)
*Keep features with correlation above threshold*
important_features = correlations[abs(correlations) > 0.1]

2- Random Forest Feature Importance
Random Forest models automatically calculate feature importance during training:

*`from sklearn.ensemble import RandomForestClassifier`*

*rf = RandomForestClassifier()*
*rf.fit(X_train, y_train)*

# Get feature importance scores
*importances = rf.feature_importances_*
# Select top N most important features
3. Recursive Feature Elimination (RFE)

Repeatedly:

Train model
Remove least important feature
Train again
Repeat

Until only the best features remain

Real Project Workflow

Clean data
        ↓
Feature Engineering
        ↓
Scale numerical features
        ↓
Feature Selection
        ↓
Train Model
        ↓
Evaluate