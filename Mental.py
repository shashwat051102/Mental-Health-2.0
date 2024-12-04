import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

train['Sleep Duration'].replace('More than 8 hours', '9-10 hours', inplace = True)
train['Sleep Duration'].replace('Less than 5 hours', '2-5 hours', inplace = True)
train['Sleep Duration'].replace('Sleep_Duration', '9-10 hours', inplace = True)
train['Sleep Duration'].replace('40-45 hours', '5-6 hours', inplace = True)
train['Sleep Duration'].replace('Moderate', '6-7 hours', inplace = True)
train['Sleep Duration'].replace('55-65 hours', '8-9 hours', inplace = True)
train['Sleep Duration'].replace('Indore', '8-9 hours', inplace = True)
train['Sleep Duration'].replace('45', '4-5 hours', inplace = True)
train['Sleep Duration'].replace('35-36 hours', '5-6 hours', inplace = True)
train['Sleep Duration'].replace('8 hours', '8-9 hours', inplace = True)
train['Sleep Duration'].replace('No', '1-2 hours', inplace = True)
train['Sleep Duration'].replace('than 5 hours', '5-6 hours', inplace = True)
train['Sleep Duration'].replace('49 hours', '7-8 hours', inplace = True)
train['Sleep Duration'].replace('Unhealthy', '2-3 hours', inplace = True)
train['Sleep Duration'].replace('Work_Study_Hours', '5-6 hours', inplace = True)
train['Sleep Duration'].replace('45-48 hours', '7-8 hours', inplace = True)
train['Sleep Duration'].replace('Pune', '7-8 hours', inplace = True)


train['Start'] =train['Sleep Duration'].str.split('-').str[0].astype(int)
train['End'] = train['Sleep Duration'].str.split('-').str[1]
train['End'] = train['End'].str.split(' ').str[0].astype(int)
train['Sleep_Duration'] = (train['End'] + train['Start']) / 2

train.drop(['Sleep Duration','Start','End'], axis = 1, inplace = True)


# Define the extended replacement dictionary
replacement_dict = {
    'Class 12': None, 'Class 11': None, '20': None, '29': None, '0': None,
    '5.61': None, '5.56': None, '7.06': None, '24': None, '8.56': None,
    '5.88': None, 'Nalini': None, 'Veda': None, 'Bhopal': None,
    'Kalyan': None, 'Unite': None, 'Vrinda': None, 'Bhavesh': None,
    'Vivaan': None, 'Brit': None, 'Ritik': None, 'Brithika': None,
    'Pihu': None, 'BB': None, 'Jhanvi': None, 'Aarav': None, 'Lata': None,
    'Marsh': None, 'Navya': None, 'Mahika': None, 'Esha': None, 'Mihir': None,
    'Advait': None, 'Degree': 'General Degree', 'BPharm': 'B.Pharm',
    'B.Sc': 'BSc', 'M_Tech': 'M.Tech', 'MTech': 'M.Tech',
    'S.Tech': None, 'H_Pharm': None, 'P.Com': None, 'P.Pharm': None,
    'LL.Com': None, 'LLCom': None, 'S.Pharm': None, 'LLBA': None,
    'M. Business Analyst': 'MBA', 'B BA': 'BBA', 'B.B.Arch': 'B.Arch',
    'MPharm': 'M.Pharm', 'HCA': None, 'ACA': 'CA', 'LCA': None, 'RCA': None,
    'K.Ed': None, 'LL B.Ed': 'B.Ed', 'M.S': 'MSc', 'E.Tech': None,
    'LHM': None, 'Doctor': 'MBBS', 'Entrepreneur': None, 'B.Student': None,
    'Working Professional': None, 'UX/UI Designer': 'Design',
    'Business Analyst': 'MBA', 'Data Scientist': 'MSc',
    'BH': None, 'BEd': 'B.Ed', 'M': 'MSc', 'M.Arch': 'M.Arch',
    'L.Ed': 'M.Ed', 'BArch': 'B.Arch', 'HR Manager': None,
    'Badhya': None, 'BPA': None, 'Plumber': None, 'B.03': None,
    'MEd': 'M.Ed', 'B': 'General Degree', 'CA': 'CA', 'CGPA': None,
    'LLTech': None, 'S.Arch': None, 'B.3.79': None, 'Mthanya': None,
    'LLS': None, 'LLEd': None, 'N.Pharm': None, 'B B.Com': 'B.Com'
}

# Apply the replacement to the 'Degree' column
train['Degree'] = train['Degree'].replace(replacement_dict)

# Replace None values with 'Other'
train['Degree'] = train['Degree'].fillna('Other')



# Define replacement dictionary for dietary habits
replacement_dict = {
    'Healthy': 2,
    'More Healthy': 1,
    'Less Healthy': 3,
    'Less than Healthy': 3,
    'No Healthy': 4,
    '2-3 hours': 'Other',
    '6-7 hours': 'Other',
    '1-2 hours': 'Other',
    'Yes': 'Other',
    'Pratham': 'Other',
    'BSc': 'Other',
    'Gender': 'Other',
    '3': 'Other',
    'Mihir': 'Other',
    '1.0': 'Other',
    'Hormonal': 'Other',
    'Electrician': 'Other',
    'nan': 'Other',
    'M.Tech': 'Other',
    'Vegas': 'Other',
    'Male': 'Other',
    'Indoor': 'Other',
    'Class 12': 'Other',
    '2': 'Other'
}

# Apply the replacements to the 'Dietary Habits' column
train['Dietary Habits'] = train['Dietary Habits'].replace(replacement_dict)

train['Dietary Habits'] = train['Dietary Habits'].replace({pd.NA: np.nan, 'nan': np.nan})

# Convert non-NaN values to numerical (if necessary)
train['Dietary Habits'] = pd.to_numeric(train['Dietary Habits'], errors='coerce')

# Optionally, replace NaNs with a specific number (e.g., -1)
train['Dietary Habits'] = train['Dietary Habits'].fillna(0)




train['Family History of Mental Illness'] = train['Family History of Mental Illness'].replace({'Yes': 1, 'No': 0})


train['Have you ever had suicidal thoughts ?'] = train['Have you ever had suicidal thoughts ?'].replace({'Yes': 1, 'No': 0})

train['Gender'] = train['Gender'].replace({'Male': 1, 'Female': 0})


train['Satisfaction'] = train['Job Satisfaction'].fillna(train['Study Satisfaction'])
train['Pressure'] = train['Academic Pressure'].fillna(train['Work Pressure'])
train.drop(columns = ['Job Satisfaction', 'Study Satisfaction', 'Academic Pressure', 'Work Pressure'], inplace = True)


train.loc[train['Working Professional or Student'] == 'Student', 'Profession'] = 'Student'
train['Profession'] = train['Profession'].fillna('Other')

train['Financial Stress'] = train['Financial Stress'].fillna(0)
train['Satisfaction'] = train['Satisfaction'].fillna(0)
train['Pressure'] = train['Pressure'].fillna(0)


# Mapping each degree to its domain
domain_mapping = {
    'B.Tech': 'Engineering and Technology',
    'M.Tech': 'Engineering and Technology',
    'BE': 'Engineering and Technology',
    'ME': 'Engineering and Technology',
    'B.Arch': 'Architecture',
    'M.Arch': 'Architecture',
    'MBBS': 'Medicine and Healthcare',
    'MD': 'Medicine and Healthcare',
    'B.Pharm': 'Medicine and Healthcare',
    'M.Pharm': 'Medicine and Healthcare',
    'MHM': 'Healthcare Management',
    'LLB': 'Law',
    'LLM': 'Law',
    'MBA': 'Management and Business',
    'BBA': 'Management and Business',
    'CA': 'Management and Business',
    'MPA': 'Management and Business',
    'BSc': 'Science',
    'MSc': 'Science',
    'BA': 'Arts and Humanities',
    'MA': 'Arts and Humanities',
    'B.Ed': 'Arts and Humanities',
    'M.Ed': 'Arts and Humanities',
    'B.Com': 'Commerce',
    'M.Com': 'Commerce',
    'Design': 'Design and Creative Fields',
    'General Degree': 'General Education',
    'Other': 'Other/Unspecified',
    'NaN': 'Other/Unspecified'
}

# Apply the mapping to a column in your DataFrame
train['Domain'] = train['Degree'].map(domain_mapping)
train['Domain'] = train['Domain'].fillna('Other/Unspecified')


train['CGPA'] = train['CGPA'].fillna(train['CGPA'].mean())

# Replace spaces with underscores in column names
train.columns = train.columns.str.replace('?', '')
train.columns = train.columns.str.replace(' ', '_')
train.columns = train.columns.str.replace('/', '_')

num_cols = train.select_dtypes(include=np.number).columns
cat_cols = train.select_dtypes(include='object').columns


num_cols_test = test.select_dtypes(include=np.number).columns
cat_cols_test = test.select_dtypes(include='object').columns


test['Sleep Duration'].replace('More than 8 hours', '9-10 hours', inplace = True)
test['Sleep Duration'].replace('Less than 5 hours', '2-5 hours', inplace = True)
test['Sleep Duration'].replace('Sleep_Duration', '9-10 hours', inplace = True)
test['Sleep Duration'].replace('60-65 hours', '8-9 hours', inplace = True)
test['Sleep Duration'].replace('Unhealthy', '2-3 hours', inplace = True)
test['Sleep Duration'].replace('55-65 hours', '8-9 hours', inplace = True)
test['Sleep Duration'].replace('Meerut', '8-9 hours', inplace = True)

test['Sleep Duration'].replace('20-21 hours', '2-3 hours', inplace = True)
test['Sleep Duration'].replace('6 hours', '8-9 hours', inplace = True)
test['Sleep Duration'].replace('45', '4-5 hours', inplace = True)
test['Sleep Duration'].replace('0', '1-2 hours', inplace = True)
test['Sleep Duration'].replace('than 5 hours', '5-6 hours', inplace = True)
test['Sleep Duration'].replace('50-75 hours', '7-8 hours', inplace = True)
test['Sleep Duration'].replace('Have_you_ever_had_suicidal_thoughts', '2-3 hours', inplace = True)
test['Sleep Duration'].replace('Work_Study_Hours', '5-6 hours', inplace = True)
test['Sleep Duration'].replace('8-89 hours', '7-8 hours', inplace = True)
test['Sleep Duration'].replace('Vivan', '7-8 hours', inplace = True)


test['Start'] = test['Sleep Duration'].str.split('-').str[0].astype(int)
test['End'] = test['Sleep Duration'].str.split('-').str[1]
test['End'] = test['End'].str.split(' ').str[0].astype(int)
test['Sleep_Duration'] = (test['End'] + test['Start']) / 2

test.drop(['Sleep Duration','Start','End'], axis = 1, inplace = True)


# Define a new replacement dictionary for this dataset
replacement_dict_test = {
    'Class 12': None, '5.65': None, '7-8 hours': None, '3.0': None,
    '8.95': None, '20': None, 'Magan': None, 'Navya': None, 'Moham': None,
    'Vibha': None, 'Bhopal': None, 'Rupak': None, 'Aadhya': None,
    'B.Sc': 'BSc', 'BPharm': 'B.Pharm', 'MPharm': 'M.Pharm', 'BTech': 'B.Tech',
    'MTech': 'M.Tech', 'BArch': 'B.Arch', 'M.Arch': 'M.Arch',
    'B_Com': 'B.Com', 'B._Pharm': 'B.Pharm', 'B Financial Analyst': None,
    'B Gender': None, 'B Study_Hours': None, 'Travel Consultant': None,
    'Mechanical Engineer': None, 'Business Analyst': 'MBA', 'B M.Com': 'M.Com',
    'B.CA': 'BCA', 'B BCA': 'BCA', 'B.Press': None, 'S.Pharm': None,
    'B.BA': 'BBA', 'B B.Tech': 'B.Tech', 'M.B.Ed': 'M.Ed', 'BEd': 'B.Ed',
    'GCA': None, 'G.Ed': None, 'RCA': None, 'PCA': None, 'J.Ed': None,
    'A.Ed': None, 'E.Ed': None, 'I.Ed': None, 'M.': None, 'K.Ed': None,
    'BH': None, 'BHCA': None, 'Degree': 'General Degree',
    'M.UI': 'Design', 'M.M.Ed': 'M.Ed', 'B.H': None, 'Advait': None,
    'Bian': None, 'Eshita': None, 'Banchal': None, 'B.Sc': 'BSc', 'B.M.Com': 'M.Com', 'B.Tech': 'B.Tech', 'MTech': 'M.Tech',
    'BPharm': 'B.Pharm', 'MPharm': 'M.Pharm', 'BArch': 'B.Arch',
    'M.Arch': 'M.Arch', 'Design': 'M.UI', 'M': None, 'B. Gender': None,
    'B.Study_Hours': None, 'General Degree': 'Other', 'Gagan': None,
    'Kavya': None, 'Vrinda': None, 'B': None
}

# Apply the replacement to the 'Degree' column
test['Degree'] = test['Degree'].replace(replacement_dict_test)

# Replace None values with 'Other'
test['Degree'] = test['Degree'].fillna('Other')


# Define the replacement dictionary
replacement_dict_test = {
    'Healthy': 2,
    'More Healthy': 1,
    'Less Healthy': 3,
    'Less than Healthy': 3,
    'No Healthy': 4,
    '2-3 hours': 'Other',
    '6-7 hours': 'Other',
    '1-2 hours': 'Other',
    'Yes': 'Other',
    'Pratham': 'Other',
    'BSc': 'Other',
    'Gender': 'Other',
    '3': 'Other',
    'Mihir': 'Other',
    '1.0': 'Other',
    'Hormonal': 'Other',
    'Electrician': 'Other',
    'nan': 'Other',
    'M.Tech': 'Other',
    'Vegas': 'Other',
    'Male': 'Other',
    'Indoor': 'Other',
    'Class 12': 'Other',
    '2': 'Other'
}

# Apply the replacement dictionary to the 'Degree' column in the test DataFrame
test['Dietary Habits'] = test['Dietary Habits'].replace(replacement_dict_test)

test['Dietary Habits'] = test['Dietary Habits'].replace({pd.NA: np.nan, 'nan': np.nan})

# Convert non-NaN values to numerical (if necessary)
test['Dietary Habits'] = pd.to_numeric(test['Dietary Habits'], errors='coerce')

# Optionally, replace NaNs with a specific number (e.g., -1)
test['Dietary Habits'] = test['Dietary Habits'].fillna(0)


test['Family History of Mental Illness'] = test['Family History of Mental Illness'].replace({'Yes': 1, 'No': 0})


test['Have you ever had suicidal thoughts ?'] = test['Have you ever had suicidal thoughts ?'].replace({'Yes': 1, 'No': 0})


test['Gender'] = test['Gender'].replace({'Male': 1, 'Female': 0})


test['Satisfaction'] = test['Job Satisfaction'].fillna(test['Study Satisfaction'])
test['Pressure'] = test['Academic Pressure'].fillna(test['Work Pressure'])
test.drop(columns = ['Job Satisfaction', 'Study Satisfaction', 'Academic Pressure', 'Work Pressure'], inplace = True)


test.loc[test['Working Professional or Student'] == 'Student', 'Profession'] = 'Student'
test['Profession'] = test['Profession'].fillna('Other')

test['Financial Stress'] = test['Financial Stress'].fillna(0)
test['Satisfaction'] = test['Satisfaction'].fillna(0)
test['Pressure'] = test['Pressure'].fillna(0)

test['CGPA'] = train['CGPA'].fillna(train['CGPA'].mean())


# Mapping each degree to its domain
domain_mapping_test = {
    'B.Tech': 'Engineering and Technology',
    'M.Tech': 'Engineering and Technology',
    'BE': 'Engineering and Technology',
    'ME': 'Engineering and Technology',
    'B.Arch': 'Architecture',
    'M.Arch': 'Architecture',
    'MBBS': 'Medicine and Healthcare',
    'MD': 'Medicine and Healthcare',
    'B.Pharm': 'Medicine and Healthcare',
    'M.Pharm': 'Medicine and Healthcare',
    'MHM': 'Healthcare Management',
    'LLB': 'Law',
    'LLM': 'Law',
    'MBA': 'Management and Business',
    'BBA': 'Management and Business',
    'CA': 'Management and Business',
    'MPA': 'Management and Business',
    'BSc': 'Science',
    'MSc': 'Science',
    'BA': 'Arts and Humanities',
    'MA': 'Arts and Humanities',
    'B.Ed': 'Arts and Humanities',
    'M.Ed': 'Arts and Humanities',
    'B.Com': 'Commerce',
    'M.Com': 'Commerce',
    'Design': 'Design and Creative Fields',
    'General Degree': 'General Education',
    'Other': 'Other/Unspecified',
    'NaN': 'Other/Unspecified'
}

# Apply the mapping to a column in your DataFrame
test['Domain'] = test['Degree'].map(domain_mapping_test)
test['Domain'] = test['Domain'].fillna('Other/Unspecified')

test.columns = test.columns.str.replace('?', '')
test.columns = test.columns.str.replace(' ', '_')
test.columns = test.columns.str.replace('/', '_')



num_cols_test = test.select_dtypes(include=np.number).columns
cat_cols_test = test.select_dtypes(include='object').columns


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

# Separate features (X) and target (y)
X = train.drop('Depression', axis=1)
y = train['Depression']

# Identify numerical and categorical columns
num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(include='object').columns

# Create transformers
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

# Create a pipeline with the preprocessor and a GradientBoostingClassifier model
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = pipeline.predict(X_val)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_val, y_pred)}")
print(classification_report(y_val, y_pred))

# Preprocess the test data using the same pipeline
test_preprocessed = preprocessor.transform(test)

# Make predictions on the preprocessed test data
test_predictions = pipeline.predict(test)

# Print or further process test predictions
test_predictions
# Create a pipeline with the preprocessor and an XGBClassifier model
xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))])

# Train the XGBClassifier model
xgb_pipeline.fit(X_train, y_train)

# Make predictions on the validation set
xgb_y_pred = xgb_pipeline.predict(X_val)

# Evaluate the XGBClassifier model
print(f"XGBClassifier Accuracy: {accuracy_score(y_val, xgb_y_pred)}")
print(classification_report(y_val, xgb_y_pred))

# Create a pipeline with the preprocessor and a CatBoostClassifier model
catboost_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('classifier', CatBoostClassifier(iterations=100, learning_rate=0.1, depth=3, random_state=42, verbose=0))])

# Train the CatBoostClassifier model
catboost_pipeline.fit(X_train, y_train)

# Make predictions on the validation set
catboost_y_pred = catboost_pipeline.predict(X_val)

# Evaluate the CatBoostClassifier model
print(f"CatBoostClassifier Accuracy: {accuracy_score(y_val, catboost_y_pred)}")
print(classification_report(y_val, catboost_y_pred))

# Make predictions on the preprocessed test data using the best model (choose one based on validation performance)
test_predictions = xgb_pipeline.predict(test)  # or catboost_pipeline.predict(test)
# Use the predictions from the RandomForestClassifier as input for the CatBoostClassifier

# Create a pipeline with the preprocessor and a RandomForestClassifier model
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

# Train the RandomForestClassifier model
rf_pipeline.fit(X_train, y_train)

# Make predictions on the validation set
rf_y_pred = rf_pipeline.predict(X_val)

# Use the predictions from the RandomForestClassifier as input features for the CatBoostClassifier
catboost_pipeline = Pipeline(steps=[('classifier', CatBoostClassifier(iterations=100, learning_rate=0.1, depth=3, random_state=42, verbose=0))])

# Train the CatBoostClassifier model on the predictions of the RandomForestClassifier
catboost_pipeline.fit(rf_y_pred.reshape(-1, 1), y_val)

# Make predictions on the test set using the RandomForestClassifier
rf_test_pred = rf_pipeline.predict(test)

# Make final predictions on the test set using the CatBoostClassifier
final_test_predictions = catboost_pipeline.predict(rf_test_pred.reshape(-1, 1))

# Print or further process final test predictions
final_test_predictions



pickle_out = open('final_model.pkl', 'wb')
pickle.dump(catboost_pipeline, pickle_out)
pickle_out.close()
