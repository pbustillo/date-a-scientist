import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix


# Import profile.csv into a DataFrame #
df = pd.read_csv("profiles.csv", low_memory = False)

# Create a column with all the joined essays which I need for the Bayes Classifier model #
essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]
# Removing the NaNs
all_essays =df[essay_cols].replace(np.nan, '', regex=True)
# Combining the essays
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)

df["all_essays"] = all_essays 

# -Lets create a DataFrame (df1) with the columns / freatures we want to use for the models- #
# -so next we can do some cleaning, numerical_mapping and normalization- #
# Note: I drop column: income because most of the responders did not respond => out of total 60.7K responders, 48.4K did not respond! #
print("48K responders did not show their income")
print(df.income.value_counts())
# Note: I drop column: ethnicity because answers are too generic #
print()
print("Check out the variaty of responses for the feature: ethnicity")
print(df.ethnicity.value_counts())
print()
columns_to_drop = ["ethnicity","income","essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9","speaks","job","last_online","location","offspring","religion","sign","pets"]
df1 = pd.DataFrame(df.drop(labels = columns_to_drop, axis=1, inplace = False))

# We need to mapp columns with str values to numerical columns  #


# Create a numerical column mapped from column: Smokes #
smokes_mapping ={"no": 0, "trying to quit": 1, "when drinking": 2, "sometimes": 3, "yes": 4, }
df1["smokes_code"] = df1.smokes.map(smokes_mapping)

# Create a numerical column mapped from column: Drinks #
drink_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
df1["drinks_code"] = df1.drinks.map(drink_mapping)

# Create a numerical column mapped from column: Drugs #
drugs_mapping = {"never": 0, "sometimes": 1, "often": 2}
df1["drugs_code"] = df1.drugs.map(drugs_mapping)

# Create a numerical column mapped from column: Education #
education_mapping ={
"working on space camp": 0,
"graduated from space camp": 1,
"space camp": 2,
"dropped out of high school": 3,
"working on high school": 4,
"graduated from high school": 5,
"high school": 6,
"dropped out of-two-year college": 7,
"working on two-year college": 8,
"graduated from two-year college": 9,
"two-year college": 10,
"dropped out of college/university": 11,
"working on college/university": 12,
"graduated from college/university": 13,
"college/university": 14,
"dropped out of masters program": 15,
"working on masters program": 16,
"graduated from masters program": 17,
"masters program": 18,
"dropped out of law school": 19,
"working on law school": 20,
"graduated from law school": 21,
"law school": 12,
"dropped out of med school": 23,
"working on med school": 24,
"graduated from med school": 25,
"med school": 26,
"dropped out of ph.d program": 27,
"working on ph.d program": 28,
"graduated from ph.d program": 29,
"ph.d program": 30}
df1["education_code"] = df1.education.map(education_mapping)

# Create a numerical column mapped from column: body_type #
body_type_mapping = {"average": 0,
"skinny": 1,
"thin": 2,
"fit": 3,
"athletic": 4,
"jacked": 5,
"a little extra": 6,
"curvy": 7,
"full figured": 8,
"overweight": 9,
"used up": 10,
"rather not say": 11}
df1["body_type_code"] = df1.body_type.map(body_type_mapping)

# Create a numerical column mapped from column: diet #
diet_mapping = {"mostly anything": 0, 
"anything": 1,
"strictly anything": 2,
"mostly vegetarian": 3,
"mostly other": 4,
"strictly vegetarian": 5,
"vegetarian": 6,
"strictly other": 7,
"mostly vegan": 8,
"other": 9,
"strictly vegan": 10,
"vegan": 11,
"mostly kosher": 12,
"strictly kosher": 13,
"strictly halal": 14,
"kosher": 15,
"halal": 16}
df1["diet_code"] = df1.diet.map(diet_mapping)

# Create a numerical column mapped from column: orientation #
orientation_mapping = {"straight": 0, 
"gay": 1,
"bisexual": 2,}
df1["orientation_code"] = df1.orientation.map(orientation_mapping)

# Create a numerical column mapped from column: status #
status_mapping = {"single": 0, 
"available": 1,
"seeing someone": 2,
"married": 3,
"unknown": 4}
df1["status_code"] = df1.status.map(status_mapping)



# I decided to drop all raws with NAN values rather than replacing the NANs with a number#
# Since there are so many NAN values in the dataframe (cutting the rows by +37K) I made a conscience trade-off#
# between a larger training data with many forced anweres (when replacing the NAN with a value) and having#
# a shorter training data but with true numerial-mapped-answeres from those who responded them all#
print()
print("Next number of missing answeres per feature both for the original Dataframe and df1")
print()
print("Original DataFrame df")
print(df.isnull().sum(axis = 0))
print()
print("For df1")
print(df1.isnull().sum(axis = 0))
print()
df1.dropna(inplace=True, how='any')

# Reset de index of the dataset #
df2 = df1.reset_index(drop = True)

# check that all columns have no missing values #
print(df2.isna().any())


# Drop out of df1 the columns that do not have numerical values #
# with the exception of "all_essays" which I need for essays_Bayes_classifer model #
columns_to_drop = ["body_type", "diet", "drinks", "drugs", "education", "orientation", "sex", "smokes", "status"]
df2.drop(labels = columns_to_drop, axis = 1, inplace = True)

# Create a DataFrame with the correlation among the features#
correlation = pd.DataFrame(df2.corr())

# Normalize the data to make sure that it all has the same weight and create a new DataFrame df3 #
df3 = df2[["age","height", "smokes_code", "drinks_code", "drugs_code", "education_code", "body_type_code", "diet_code", "orientation_code", "status_code"]]
x = df3.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df3 = pd.DataFrame(x_scaled, columns=df3.columns)



# Prediction of Age -- Using Regression Analysis #
# Based on data correlation: smokes_code, drinks_code, drugs_code, education_code#
# Select the feature and target variables#
feature1 = df3[["smokes_code", "drinks_code", "drugs_code", "education_code"]]
target1 = df2["age"] # Dont want the target to be normalized #
# Split the data into Training and Testing Sets #
X_train1, X_test1, y_train1, y_test1 = train_test_split(feature1, target1, test_size = 0.2, random_state = 1)
# Create and train the model #
model1 = LinearRegression()
model1.fit(X_train1, y_train1)

# Lets predict the Age #
y_predicted1 = model1.predict(X_test1)
print("R´^2 : Regression analysis _ Age")
print()
print("The R^2 for the Train data is")
print(model1.score(X_train1, y_train1))
print("The R^2 for the test data is")
print(model1.score(X_test1, y_test1))
print()
plt.figure(1)
plt.scatter(y_test1, y_predicted1)
plt.xlabel("Age")
plt.ylabel("Predicted Age")
plt.ylim(0,70)
plt.show()

#prediction of Age with education -- using Regression Analysis #

feature2 = df3[["education_code"]]
target2 = df2["age"] # Dont want the target to be normalized #
# Split the data into Training and Testing Sets #
X_train2, X_test2, y_train2, y_test2 = train_test_split(feature2, target2, test_size = 0.2, random_state = 1)
# Create and train the model #
model2 = LinearRegression()
model2.fit(X_train2, y_train2)

# Lets predict the Age #
y_predicted2 = model2.predict(X_test2)
print("R´^2 : Regression analysis _ Age / education")
print()
print("The R^2 for the Train data is")
print()
print("The R^2 for the Train data is")
print(model2.score(X_train2, y_train2))
print("The R^2 for the test data is")
print(model2.score(X_test2, y_test2))
print()
plt.figure(2)
plt.scatter(y_test2, y_predicted2)
plt.xlabel("Age")
plt.ylabel("Predicted Age")
plt.ylim(0,70)
plt.show()



# Prediction of Drugs usage -- Using Regression Analysis #
# Features based on correlation: age, smokes_code, drinks_code, education_code, orientation_code #
# Select the feature and target variables #
feature3 = df3[["age", "drinks_code", "smokes_code", "education_code"]]
target3 = df2["drugs_code"] # Dont want the target to be normalized #
# Split the data into Training and Testing Sets #
X_train3, X_test3, y_train3, y_test3 = train_test_split(feature3, target3, test_size = 0.2, random_state = 1)
# Create and train the model #
model3 = LinearRegression()
model3.fit(X_train3, y_train3)

# Lets predict the Drug usage __ Using Regression Analysis #
y_predicted3 = model3.predict(X_test3)
print("R´^2 : Regression analysis _ Drugs usage")
print()
print("The R^2 for the Train data is")
print()
print("The R^2 for the Train data is")
print(model3.score(X_train3, y_train3))
print("The R^2 for the test data is")
print(model3.score(X_test3, y_test3))
print()
plt.figure(3)
plt.scatter(y_test3, y_predicted3)
plt.xlabel("Drugs usage")
plt.ylabel("Predicted Drugs usage")
plt.ylim(0,2)
plt.show()

# What happens if we use KMeans to classify a person drugs usage group #
# Finding patterns and structure in data with no labeled answers #
# Lets start with 3 clusters: one for each group of drugs_code responses #
classifier = KMeans (n_clusters = 3)
# To fit the model to samples #
classifier.fit(df3[["drinks_code", "smokes_code", "education_code"]])
# To determine the labels of samples #
labels = classifier.predict(df3[["drinks_code", "smokes_code", "education_code"]])
# Create a Dataframe with two columns: labels for each row predicted by the model and the target ...#
# ... or the row's drugs_code #
results = pd.DataFrame({"labels": labels})
results["target"] = df3["drugs_code"]
# Have a cross-view of the drugs_code groups assigned to each cluster label #
ct = pd.crosstab(results["labels"], results["target"])
print("How the rows were assigned to each cluster")
print()
print(ct)
print()
print("the model inertia is:")
print()
print(classifier.inertia_)
print()




# Can we predict orientation using the writing content of the essays #
# Classifying Using Lenguage: Naive Bayes Classifier #
# Make a list out of the all_essays column in df2
essays_to_classify = list(df2["all_essays"])
# Make a list ouf the all responses for orientation #
labels1 = list(df2["orientation_code"])
train_data, test_data, train_labels, test_labels = train_test_split(essays_to_classify, labels1, test_size = 0.2, random_state = 1)
print(len(train_data))
print(len(test_data))
print()
count_vectors = CountVectorizer()
count_vectors.fit(train_data)
train_counts = count_vectors.transform(train_data)
test_counts = count_vectors.transform(test_data)
print(train_data[3])
print(train_counts[3])
print()

classifier2 = MultinomialNB()
classifier2.fit(train_counts, train_labels)
predictions = classifier2.predict(test_counts)
print("The model accuracy score is:")
print(accuracy_score(test_labels, predictions))
print()
print( "The model confusion metrix is:")
print(confusion_matrix(test_labels, predictions))


        

























