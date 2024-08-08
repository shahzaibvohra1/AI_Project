import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
datafm = pd.read_csv("/Users/artyom/Downloads/Killed_and_Seriously_Injured.csv")

datafm.describe()



datafm.info()


# Determine the number of numerical columns
numerical_columns = datafm.select_dtypes(include=['float64','int64']).columns
num_numerical_columns = len(numerical_columns)

# Determine the number of categorical columns
categorical_columns = datafm.select_dtypes(include=['object']).columns
num_categorical_columns = len(categorical_columns)

# Print the results
print(f'Number of numerical columns: {num_numerical_columns}')
print(f'Number of categorical columns: {num_categorical_columns}')


datafm.head()
data = datafm[['X','Y','LATITUDE','LONGITUDE']]
datafm['X']


# Check for missing values
missing_values = datafm.isnull().sum()
missing_values = missing_values[missing_values > 0]

# Plot missing values
plt.figure(figsize=(12, 6))
missing_values.plot(kind='bar', color='salmon')
plt.title('Missing Values by Column')
plt.xlabel('Columns')
plt.ylabel('Count of Missing Values')
plt.show()


import geopandas as gpd
from shapely.geometry import Point

geometry = [Point(xy) for xy in zip(datafm['LONGITUDE'], datafm['LATITUDE'])]
geo_df = gpd.GeoDataFrame(datafm, geometry=geometry)

# Filter data based on ACCLASS
non_fatal_accidents = geo_df[geo_df['ACCLASS'] != 'Fatal']
fatal_accidents = geo_df[geo_df['ACCLASS'] == 'Fatal']


# Plot the data
fig, ax = plt.subplots(figsize=(12, 10))
base = geo_df.plot(ax=ax, color='none', edgecolor='black')
non_fatal_accidents.plot(ax=base, marker='o', color='green', markersize=5, alpha=0.5, label='Non-Fatal')
fatal_accidents.plot(ax=base, marker='o', color='red', markersize=5, alpha=0.5, label='Fatal')


plt.title('Geospatial Plot of Accidents: Red for Fatal, Green for Non-Fatal')
plt.legend()
plt.show()



datafm = datafm.dropna(subset=['ACCLASS'])

plt.figure(figsize=(8, 6))
sns.countplot(x='ACCLASS', data=datafm, palette='viridis')

# Get value counts and annotate bars
value_counts = datafm['ACCLASS'].value_counts()
for i, count in enumerate(value_counts):
    plt.text(i, count + 0.1, str(count), ha='center')

plt.title('Class Distribution of ACCLASS')
plt.xlabel('ACCLASS')
plt.ylabel('Count')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()


datafm['ACCLASS'] = np.where(datafm['ACCLASS'] == 'Property Damage O', 'Non-Fatal Injury', datafm['ACCLASS'])


datafm = datafm.dropna(subset=['ACCLASS'])

plt.figure(figsize=(8, 6))
sns.countplot(x='ACCLASS', data=datafm, palette='viridis')

# Get value counts and annotate bars
value_counts = datafm['ACCLASS'].value_counts()
for i, count in enumerate(value_counts):
    plt.text(i, count + 0.1, str(count), ha='center')

plt.title('Class Distribution of ACCLASS')
plt.xlabel('ACCLASS')
plt.ylabel('Count')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()


datafm['datetime'] = pd.to_datetime(datafm['DATE'], format='%Y/%m/%d %H:%M:%S%z')

# Extract year and month
datafm['YEAR'] = datafm['datetime'].dt.year
datafm['MONTH'] = datafm['datetime'].dt.month

datafm.describe()

#Number of Unique accidents by Year
Num_accident = datafm.groupby('YEAR')['ACCNUM'].nunique()
plt.figure(figsize=(12,6))
plt.title("Accidents caused in different years")
plt.ylabel('Number of Accidents (ACCNUM)')

ax = plt.gca()
ax.tick_params(axis='x', colors='blue')
ax.tick_params(axis='y', colors='red')
my_colors = list('rgbkymc')   #red, green, blue, black, etc.
Num_accident.plot(
    kind='bar', 
    color='blue',
    edgecolor='black'
)
#Num_accident.plot(kind='bar',color= my_colors)
plt.show()

datafm["YEAR"]



#Looking at area where accident happens

Region_KSI_CLEAN = datafm['DISTRICT'].value_counts()
plt.figure(figsize=(12,6))
plt.ylabel('Number of Accidents')
Region_KSI_CLEAN.plot(kind='bar',color=list('rgbkmc') )
plt.show()




Hood_KSI_CLEAN = datafm['NEIGHBOURHOOD_140'].value_counts()
plt.figure(figsize=(12,6))
plt.ylabel('Number of Accidents')
Hood_KSI_CLEAN.nlargest(20).plot(kind='bar',color=list('rgbkmc') )
plt.show()



Hood_KSI_CLEAN = datafm['NEIGHBOURHOOD_158'].value_counts()
plt.figure(figsize=(12,6))
plt.ylabel('Number of Accidents')
Hood_KSI_CLEAN.nlargest(50).plot(kind='bar',color=list('rgbkmc') )
plt.show()


# Identify categorical columns (modify based on your dataset)
selected_columns = ['SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL']
categorical_cols = datafm[selected_columns]
print(datafm[selected_columns])
datafm[selected_columns] = datafm[selected_columns].notna() & datafm[selected_columns].ne("YES")
datafm[selected_columns] = datafm[selected_columns].astype(int)

datafm.head()
datafm["YEAR"]


## Driving condition VS accident #
## creating a pivot table for accidents causing by 'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL'  for EDA.
KSI_pivot_cause = datafm.pivot_table(index='YEAR', 
                           values = ['SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL'],
                           aggfunc=np.sum,
                           margins = True,
                           margins_name = 'Total Under Category')
fig, ax1 = plt.subplots(figsize=(8,8))
KSI_pivot_cause.iloc[11].plot(kind='pie', ax=ax1, autopct='%3.1f%%',fontsize=10)
ax1.set_ylabel('')
ax1.set_xlabel('Driving condition VS Accidents in Ontario in last 18 years(%age)',fontsize=10)


KSI_pivot_cause = KSI_pivot_cause.reset_index()
# KSI_pivot_cause = KSI_pivot_cause.drop(columns=["level_0","index"])
KSI_pivot_cause = KSI_pivot_cause[KSI_pivot_cause['YEAR'] != 'Total Under Category']

# Plotting
KSI_pivot_cause.plot(x='YEAR', kind='bar', stacked=False, figsize=(12, 8))

# Adding title and labels
plt.title('Frequency of Each Class by Year')
plt.xlabel('Year')
plt.ylabel('Frequency')

# Adding legend
plt.legend(title='Class')

# Display the plot
plt.tight_layout()
plt.show()



selected_columns = [ 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK', 'TRSN_CITY_VEH', 'EMERG_VEH' ]
datafm[selected_columns] = datafm[selected_columns].notna() & datafm[selected_columns].ne("YES")
datafm[selected_columns] = datafm[selected_columns].astype(int)
KSI_pivot_Types = datafm.pivot_table(index='YEAR', 
                           values = [ 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK', 'TRSN_CITY_VEH', 'EMERG_VEH'],
                           aggfunc=np.sum,
                           margins = True,
                           margins_name = 'Total Under Category')

fig, ax1 = plt.subplots(figsize=(8,8))
KSI_pivot_Types.iloc[11].plot(kind='pie', ax=ax1, autopct='%3.1f%%',fontsize=10)
ax1.set_ylabel('')
ax1.set_xlabel('Vechile type VS Accidents in Ontario in last 18 years(%age)',fontsize=10)



KSI_pivot_Types = KSI_pivot_Types.reset_index()
# KSI_pivot_Types = KSI_pivot_Types.drop(columns=["level_0","index"])
KSI_pivot_Types = KSI_pivot_Types[KSI_pivot_Types['YEAR'] != 'Total Under Category']

# Plotting
KSI_pivot_Types.plot(x='YEAR', kind='bar', stacked=False, figsize=(12, 8))

# Adding title and labels
plt.title('Frequency of Each Class by Year')
plt.xlabel('Year')
plt.ylabel('Frequency')

# Adding legend
plt.legend(title='Class')

# Display the plot
plt.tight_layout()
plt.show()


## Victims VS accident #
## creating a pivot table for Victims by 'CYCLIST','PEDESTRIAN','PASSENGER'
# 
selected_columns = [ 'CYCLIST','PEDESTRIAN','PASSENGER' ]
datafm[selected_columns] = datafm[selected_columns].notna() & datafm[selected_columns].ne("YES")
datafm[selected_columns] = datafm[selected_columns].astype(int)

KSI_pivot_CPP = datafm.pivot_table(index='YEAR', 
                           values = [ 'CYCLIST','PEDESTRIAN','PASSENGER' ],
                           aggfunc=np.sum,
                           margins = True,
                           margins_name = 'Total Under Category')
fig, ax1 = plt.subplots(figsize=(8,8))
KSI_pivot_CPP.iloc[11].plot(kind='pie', ax=ax1, autopct='%3.1f%%',fontsize=10)
ax1.set_ylabel('')
ax1.set_xlabel('Victims VS Accidents in Ontario in last 18 years(%age)',fontsize=10)


# Group by 'YEAR', 'MONTH', and 'ACCLASS' and count the values
data_grouped = datafm.groupby(['YEAR', 'MONTH', 'ACCLASS']).size().reset_index(name='count')

# Filter data for 'fatal' accidents and pivot
fatal_data = data_grouped[data_grouped['ACCLASS'] == 'Fatal'].pivot(index='MONTH', columns='YEAR', values='count')
fatal_data = fatal_data.fillna(0).astype(int)  # Fill NaN with 0 and convert to int
plt.figure(figsize=(12,6))
sns.heatmap(fatal_data, center=fatal_data.loc[1, 2006], annot=True, fmt="d", cmap="YlGnBu")
plt.show()



datafm.info()


visibility_acclass_counts = datafm.groupby(['VISIBILITY', 'ACCLASS']).size().reset_index(name='count')

# Set up the matplotlib figure
plt.figure(figsize=(10, 6))

# Create a bar plot using seaborn
sns.barplot(data=visibility_acclass_counts, x='VISIBILITY', y='count', hue='ACCLASS')

# Add titles and labels
plt.title('Frequency of Fatal and Non-Fatal Incidents by Visibility')
plt.xlabel('Visibility')
plt.ylabel('Frequency')
plt.legend(title='ACCLASS')

# Show the plot
plt.show()


visibility_acclass_counts = datafm.groupby(['LIGHT', 'ACCLASS']).size().reset_index(name='count')

# Set up the matplotlib figure
plt.figure(figsize=(10, 6))

# Create a bar plot using seaborn
sns.barplot(data=visibility_acclass_counts, x='LIGHT', y='count', hue='ACCLASS')

# Add titles and labels
plt.title('Frequency of Fatal and Non-Fatal Incidents by LIGHT')
plt.xlabel('LIGHT')
plt.ylabel('Frequency')
plt.legend(title='ACCLASS')

# Show the plot
plt.show()



visibility_acclass_counts = datafm.groupby(['ACCLOC', 'ACCLASS']).size().reset_index(name='count')

# Set up the matplotlib figure
plt.figure(figsize=(10, 6))

# Create a bar plot using seaborn
sns.barplot(data=visibility_acclass_counts, x='ACCLOC', y='count', hue='ACCLASS')

# Add titles and labels
plt.title('Frequency of Fatal and Non-Fatal Incidents by ACCLOC')
plt.xlabel('ACCLOC')
plt.ylabel('Frequency')
plt.legend(title='ACCLASS')

# Show the plot
plt.show()
datafm.ACCLOC.unique()



visibility_acclass_counts = datafm.groupby(['ROAD_CLASS', 'ACCLASS']).size().reset_index(name='count')

# Set up the matplotlib figure
plt.figure(figsize=(10, 6))

# Create a bar plot using seaborn
sns.barplot(data=visibility_acclass_counts, x='ROAD_CLASS', y='count', hue='ACCLASS')

# Add titles and labels
plt.title('Frequency of Fatal and Non-Fatal Incidents by ROAD_CLASS')
plt.xlabel('ROAD_CLASS')
plt.ylabel('Frequency')
plt.legend(title='ACCLASS')

# Show the plot
plt.show()
datafm.ACCLOC.unique()


visibility_acclass_counts = datafm.groupby(['DIVISION', 'ACCLASS']).size().reset_index(name='count')

# Set up the matplotlib figure
plt.figure(figsize=(10, 6))

# Create a bar plot using seaborn
sns.barplot(data=visibility_acclass_counts, x='DIVISION', y='count', hue='ACCLASS')

# Add titles and labels
plt.title('Frequency of Fatal and Non-Fatal Incidents by DIVISION')
plt.xlabel('DIVISION')
plt.ylabel('Frequency')
plt.legend(title='ACCLASS')

# Show the plot
plt.show()

      



visibility_acclass_counts = datafm.groupby(['TRAFFCTL', 'ACCLASS']).size().reset_index(name='count')

# Set up the matplotlib figure
plt.figure(figsize=(15, 6))

# Create a bar plot using seaborn
sns.barplot(data=visibility_acclass_counts, x='TRAFFCTL', y='count', hue='ACCLASS')

# Add titles and labels
plt.title('Frequency of Fatal and Non-Fatal Incidents by TRAFFCTL')
plt.xlabel('TRAFFCTL')
plt.ylabel('Frequency')
plt.legend(title='ACCLASS')

# Show the plot
plt.show()




visibility_acclass_counts = datafm.groupby(['IMPACTYPE', 'ACCLASS']).size().reset_index(name='count')

# Set up the matplotlib figure
plt.figure(figsize=(15, 6))

# Create a bar plot using seaborn
sns.barplot(data=visibility_acclass_counts, x='IMPACTYPE', y='count', hue='ACCLASS')

# Add titles and labels
plt.title('Frequency of Fatal and Non-Fatal Incidents by IMPACTYPE')
plt.xlabel('IMPACTYPE')
plt.ylabel('Frequency')
plt.legend(title='ACCLASS')

# Show the plot
plt.show()



visibility_acclass_counts = datafm.groupby(['INVTYPE', 'ACCLASS']).size().reset_index(name='count')

# Set up the matplotlib figure
plt.figure(figsize=(20, 6))

# Create a bar plot using seaborn
sns.barplot(data=visibility_acclass_counts, x='INVTYPE', y='count', hue='ACCLASS')

# Add titles and labels
plt.title('Frequency of Fatal and Non-Fatal Incidents by INVTYPE')
plt.xlabel('INVTYPE')
plt.ylabel('Frequency')
plt.legend(title='ACCLASS')

# Show the plot
plt.show()



visibility_acclass_counts = datafm.groupby(['INVAGE', 'ACCLASS']).size().reset_index(name='count')

# Set up the matplotlib figure
plt.figure(figsize=(20, 6))

# Create a bar plot using seaborn
sns.barplot(data=visibility_acclass_counts, x='INVAGE', y='count', hue='ACCLASS')

# Add titles and labels
plt.title('Frequency of Fatal and Non-Fatal Incidents by INVAGE')
plt.xlabel('INVAGE')
plt.ylabel('Frequency')
plt.legend(title='ACCLASS')

# Show the plot
plt.show()


visibility_acclass_counts = datafm.groupby(['INITDIR', 'ACCLASS']).size().reset_index(name='count')

# Set up the matplotlib figure
plt.figure(figsize=(20, 6))

# Create a bar plot using seaborn
sns.barplot(data=visibility_acclass_counts, x='INITDIR', y='count', hue='ACCLASS')

# Add titles and labels
plt.title('Frequency of Fatal and Non-Fatal Incidents by INITDIR')
plt.xlabel('INITDIR')
plt.ylabel('Frequency')
plt.legend(title='ACCLASS')

# Show the plot
plt.show()


visibility_acclass_counts = datafm.groupby(['VEHTYPE', 'ACCLASS']).size().reset_index(name='count')

# Set up the matplotlib figure
plt.figure(figsize=(35, 10))

# Create a bar plot using seaborn
sns.barplot(data=visibility_acclass_counts, x='VEHTYPE', y='count', hue='ACCLASS')

# Add titles and labels
plt.title('Frequency of Fatal and Non-Fatal Incidents by VEHTYPE')
plt.xlabel('VEHTYPE')
plt.ylabel('Frequency')
plt.legend(title='ACCLASS')

# Show the plot
plt.show()





visibility_acclass_counts = datafm.groupby(['MANOEUVER', 'ACCLASS']).size().reset_index(name='count')

# Set up the matplotlib figure
plt.figure(figsize=(20, 10))

# Create a bar plot using seaborn
sns.barplot(data=visibility_acclass_counts, x='MANOEUVER', y='count', hue='ACCLASS')

# Add titles and labels
plt.title('Frequency of Fatal and Non-Fatal Incidents by MANOEUVER')
plt.xlabel('MANOEUVER')
plt.ylabel('Frequency')
plt.legend(title='ACCLASS')

# Show the plot
plt.show()






visibility_acclass_counts = datafm.groupby(['DRIVACT', 'ACCLASS']).size().reset_index(name='count')

# Set up the matplotlib figure
plt.figure(figsize=(25, 6))

# Create a bar plot using seaborn
sns.barplot(data=visibility_acclass_counts, x='DRIVACT', y='count', hue='ACCLASS')

# Add titles and labels
plt.title('Frequency of Fatal and Non-Fatal Incidents by DRIVACT')
plt.xlabel('DRIVACT')
plt.ylabel('Frequency')
plt.legend(title='ACCLASS')

# Show the plot
plt.show()








visibility_acclass_counts = datafm.groupby(['DRIVCOND', 'ACCLASS']).size().reset_index(name='count')

# Set up the matplotlib figure
plt.figure(figsize=(25, 6))

# Create a bar plot using seaborn
sns.barplot(data=visibility_acclass_counts, x='DRIVCOND', y='count', hue='ACCLASS')

# Add titles and labels
plt.title('Frequency of Fatal and Non-Fatal Incidents by DRIVCOND')
plt.xlabel('DRIVCOND')
plt.ylabel('Frequency')
plt.legend(title='ACCLASS')

# Show the plot
plt.show()







visibility_acclass_counts = datafm.groupby(['PEDESTRIAN', 'ACCLASS']).size().reset_index(name='count')

# Set up the matplotlib figure
plt.figure(figsize=(25, 6))

# Create a bar plot using seaborn
sns.barplot(data=visibility_acclass_counts, x='PEDESTRIAN', y='count', hue='ACCLASS')

# Add titles and labels
plt.title('Frequency of Fatal and Non-Fatal Incidents by PEDESTRIAN')
plt.xlabel('PEDESTRIAN')
plt.ylabel('Frequency')
plt.legend(title='ACCLASS')

# Show the plot
plt.show()







visibility_acclass_counts = datafm.groupby(['NEIGHBOURHOOD_158', 'ACCLASS']).size().reset_index(name='count')

# Set up the matplotlib figure
plt.figure(figsize=(25, 6))

# Create a bar plot using seaborn
sns.barplot(data=visibility_acclass_counts, x='NEIGHBOURHOOD_158', y='count', hue='ACCLASS')

# Add titles and labels
plt.title('Frequency of Fatal and Non-Fatal Incidents by NEIGHBOURHOOD_158')
plt.xlabel('NEIGHBOURHOOD_158')
plt.ylabel('Frequency')
plt.legend(title='ACCLASS')

# Show the plot
plt.show()



from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier


dropColumns = ["datetime","DATE","TIME","X","Y","OFFSET", "FATAL_NO", "PEDTYPE", "PEDACT", "PEDCOND",
               "CYCLISTYPE", "CYCACT", "CYCCOND","DISABILITY", "LATITUDE", "LONGITUDE",
                "STREET1","STREET2","ACCNUM","LIGHT","INJURY","DIVISION", "MANOEUVER",
                 "INITDIR", "INVAGE", "VEHTYPE","DRIVACT", "DRIVCOND", "HOOD_158", 
                 "NEIGHBOURHOOD_158", "HOOD_140","OBJECTID","INDEX_" ]

datafm = datafm.drop(columns=dropColumns)

# Encode target variable 'ACCLASS' using LabelEncoder
label_encoder = LabelEncoder()
datafm['ACCLASS'] = label_encoder.fit_transform(datafm['ACCLASS'])

# Separate features (X) and target (y)
X = datafm.drop('ACCLASS', axis=1)
y = datafm['ACCLASS']

# Define categorical columns for OneHotEncoding
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()

# Define preprocessing steps for the pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_columns)
    ],
    remainder='passthrough'
)

# Create pipeline including feature selection and model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selection', SelectKBest(score_func=chi2, k=20)),  # Select top 20 features using chi2
    ('classifier', RandomForestClassifier())  # RandomForestClassifier as an example
])

# Perform stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Fit pipeline on training data
pipeline.fit(X_train, y_train)
