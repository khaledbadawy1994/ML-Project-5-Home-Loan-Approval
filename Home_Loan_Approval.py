# ML-Project-5-Home-Loan-Approval

#Loading Packages

import pandas as pdimport numpy as np                     # For mathematical calculationsimport seaborn as sns                  # For data visualizationimport matplotlib.pyplot as plt        # For plotting graphs%matplotlib inlineimport warnings                        # To ignore any warningswarnings.filterwarnings("ignore")import plotly.express as pximport plotly.graph_objects as gofrom plotly.subplots import make_subplots

from google.colab import drivedrive.mount('/content/drive')

Mounted at /content/drive
train=pd.read_csv("/content/drive/MyDrive/loan_sanction_train.csv")test=pd.read_csv("/content/drive/MyDrive/loan_sanction_test.csv")

train.columns

test.columns

# Print data types for each variable
train.dtypes

train.shape, test.shape

train['Loan_Status'].value_counts()

# Normalize can be set to True to print proportions instead of number
train['Loan_Status'].value_counts(normalize=True)

plt.figure(figsize=(8, 6))  # Adjust the figure size as needed

# Plotting the bar chart
train['Loan_Status'].value_counts().plot.bar(color=['green', 'red'])  # Customize colors if desired

# Adding labels and title
plt.title('Loan Approval Status')
plt.xlabel('Loan Status')
plt.ylabel('Count')

# Display the plot
plt.show()

422(around 69%) people out of 614 got the approval.

pd.crosstab(train['Credit_History'], train['Loan_Status']).plot.bar(color = ['orangered', 'palegreen'], figsize = [15,4])
plt.xticks(rotation = 45)
plt.title('Loan Status / Credit History')
plt.grid()
plt.ylabel('Number of loans');

pd.crosstab(train['Credit_History'], train['Loan_Status'])

train.head()

train.info

# Checking Statistical data
train.describe()

Note:

#Minimum Salary of Applicant is 150 Maximum Salary of Applicant is 81000 Average Salary of Applicant is 5403 Mean loan amt is 146 Mean loan term is 342

train.isnull()

# Cheking if there is null value

train.isnull().sum()

train_cleaned =train.dropna()
train_cleaned

# Data Cleaning

train_cleaned.isnull().sum()

# Drop rows with any missing values
train_drop_row = train.dropna()
train_drop_row

# Drop columns with any missing values
train_drop_columns = train.dropna(axis=1)
train_drop_columns

# Drop rows only if all columns have missing values
train_drop_row_all = train.dropna(how="all")
train_drop_row_all

#train

train.duplicated().any()

#There are no duplicated rows

# Remove NaN values in a specific column
train = train.dropna(subset=['LoanAmount'])
train

has_nan = train.isna().any().any().sum()

train_without_nan_rows = train.dropna(axis=0)
train

#Missing Value and Outlier Treatment

train.isnull().sum()

#For numerical variables: imputation using mean or median For categorical variables: imputation using mode

train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
train['Loan_Amount_Term'].value_counts()

#It can be seen that in the loan amount term variable, the value of 360 is repeated the most. So we will replace the missing values in this variable using the mode of this variable.

train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)
train.isnull().sum()

test['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

#Outlier Treatment

train['LoanAmount_log'] = np.log(train['LoanAmount'])
train['LoanAmount_log'].hist(bins=20)
test['LoanAmount_log'] = np.log(test['LoanAmount'])

#Now the distribution looks much closer to normal and the effect of extreme values has been significantly subsided.

plt.figure(figsize= [15,5])
sns.boxplot(data =train)
plt.style.use('ggplot')
plt.grid()

def detect_outlier(col):
    '''
    function to detect outliers (using z score)
    '''
    outliers = []
    threshold = 3                       # seuil standard
    mean = np.mean(col)
    std = np.std(col)

    for i in col :
        z_score = (i - mean) / std            # formule z-score
        if np.abs(z_score) > threshold :
            outliers.append(i)
    return outliers
    
detect_outlier(train['ApplicantIncome'])   # drop 81000, 63000

detect_outlier(train['CoapplicantIncome'])    # drop 41000

train.isnull().sum()

#dependents column
train['Dependents'] = train['Dependents'].fillna('0')
train.isna().sum()

#Married column
train = train[train['Married'].notna()]
train.isna().sum()

#Self_employed column
train['Self_Employed'] = train['Self_Employed'].fillna('No')
train.isna().sum()

#Loan amount column
train['LoanAmount'] = train['LoanAmount'].fillna(train['LoanAmount'].median())
train.isna().sum()

#Loan amount term column
train['Loan_Amount_Term'] = train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].min())
train.isna().sum()

#credit history column

train['Credit_History'] = train['Credit_History'].fillna(0)

train.isna().sum()

train.shape

#Now that all the null values are succesfully removed, lets have a look at the columns individually.

train['Gender'].unique()

train['Married'].unique()

train['Dependents'].unique()

train['Education'].unique()

train['Self_Employed'].unique()

train['Credit_History'].unique()

train['Property_Area'].unique()

train['Loan_Status'].unique()

train['Dependents'].unique()

#Credit History column values can be changed to int.

train['Credit_History'] = train['Credit_History'].astype(int)
train['Credit_History'].dtype

train.head()

#Data Visualization

# Create a DataFrame with the counts and percentages
gender_counts = train['Gender'].value_counts(normalize=True).reset_index()
gender_counts.columns = ['Gender', 'Percentage']

# Create an advanced bar chart using Plotly Express
fig = px.bar(gender_counts,
             x='Gender',
             y='Percentage',
             text='Percentage',
             labels={'Percentage': 'Percentage'},
             color='Gender',
             color_discrete_sequence=['blue', 'orange'],  # Customize colors if desired
             title='Gender Distribution',
             template = 'plotly_dark',
             hover_data={'Gender': False, 'Percentage': ':.2%'},
             width=800, height=500)

# Customize layout
fig.update_layout(
    xaxis_title='Gender',
    yaxis_title='Percentage',
    showlegend=False,  # Hide legend for this specific chart
    bargap=0.1,  # Set gap between bars
)

# Display the plot
fig.show()

# Create a DataFrame with the counts and percentages
married_counts = train['Married'].value_counts(normalize=True).reset_index()
married_counts.columns = ['Married', 'Percentage']

# Create an advanced bar chart using Plotly Express
fig = px.bar(married_counts,
             x='Married',
             y='Percentage',
             text='Percentage',
             labels={'Percentage': 'Percentage'},
             color='Married',
             color_discrete_sequence=['#636EFA', '#EF553B'],  # Customize colors if desired
             title='Marital Status Distribution',
             hover_data={'Married': False, 'Percentage': ':.2%'},
             width=800, height=500)

# Customize layout
fig.update_layout(
    xaxis_title='Marital Status',
    yaxis_title='Percentage',
    showlegend=False,  # Hide legend for this specific chart
    bargap=0.1,  # Set gap between bars
    template='plotly_dark',  # Set dark template
)

# Display the plot
fig.show()

# Create a DataFrame with the counts and percentages
self_employed_counts = train['Self_Employed'].value_counts(normalize=True).reset_index()
self_employed_counts.columns = ['Self_Employed', 'Percentage']

# Create an advanced bar chart using Plotly Express
fig = px.bar(self_employed_counts,
             x='Self_Employed',
             y='Percentage',
             text='Percentage',
             labels={'Percentage': 'Percentage'},
             color='Self_Employed',
             color_discrete_sequence=['#1f77b4', '#ff7f0e'],  # Customize colors if desired
             title='Self Employment Status Distribution',
             hover_data={'Self_Employed': False, 'Percentage': ':.2%'},
             width=800, height=500)

# Set dark template
fig.update_layout(template='plotly_dark')

# Customize layout
fig.update_layout(
    xaxis_title='Self Employed',
    yaxis_title='Percentage',
    showlegend=False,  # Hide legend for this specific chart
    bargap=0.1,  # Set gap between bars
)

# Display the plot
fig.show()

# Create a DataFrame with the counts and percentages
credit_history_counts = train['Credit_History'].value_counts(normalize=True).reset_index()
credit_history_counts.columns = ['Credit_History', 'Percentage']

# Create an advanced bar chart using Plotly Express
fig = px.bar(credit_history_counts,
             x='Credit_History',
             y='Percentage',
             text='Percentage',
             labels={'Percentage': 'Percentage'},
             color='Credit_History',
             color_discrete_sequence=['red', 'green'],  # Customize colors if desired
             title='Credit History Distribution',
             hover_data={'Credit_History': False, 'Percentage': ':.2%'},
             width=800, height=500)

# Use a dark template
fig.update_layout(template='plotly_dark')

# Customize layout
fig.update_layout(
    xaxis_title='Credit History',
    yaxis_title='Percentage',
    showlegend=False,  # Hide legend for this specific chart
    bargap=0.1,  # Set gap between bars
    )

# Display the plot
fig.show()

🕵️Observations:

80% of applicants in the dataset are male. Around 65% of the applicants in the dataset are married. About 15% of applicants in the dataset are self-employed. About 84% of applicants have repaid their debts.

import plotly.subplots as sp
import plotly.graph_objects as go

# Assuming 'train' is your DataFrame

# Create subplots
fig = sp.make_subplots(rows=1, cols=3, subplot_titles=['Dependents', 'Education', 'Property Area'])

# Plot Dependents distribution
fig.add_trace(go.Bar(x=train['Dependents'].value_counts(normalize=True).index,
                     y=train['Dependents'].value_counts(normalize=True),
                     name='Dependents'),
              row=1, col=1)

# Plot Education distribution
fig.add_trace(go.Bar(x=train['Education'].value_counts(normalize=True).index,
                     y=train['Education'].value_counts(normalize=True),
                     name='Education'),
              row=1, col=2)

# Plot Property Area distribution
fig.add_trace(go.Bar(x=train['Property_Area'].value_counts(normalize=True).index,
                     y=train['Property_Area'].value_counts(normalize=True),
                     name='Property Area'),
              row=1, col=3)

# Update layout
fig.update_layout(height=500, showlegend=False, title_text="Categorical Distributions")

# Display the plot
fig.show()

🕵️Observations:

Most of the applicants don’t have dependents. About 78% of the applicants are graduates. Most of the applicants are from semi-urban areas.

cat_features = train.select_dtypes('object').columns
fig,ax = plt.subplots(2,4,figsize=(20,12))

for i,feature in enumerate(cat_features):
    order = (
        train[feature]
        .value_counts(normalize = True )
        .sort_values(ascending=False).index
    )
    sns.countplot(data=train,x=feature,ax=ax[i//4,i%4],order = order)
    ax[i//4,i%4].set_title(f'{feature} Distribution')

#General notes:

most of the applicants are males. most of the applicants are married. most of the applicants have no dependents. most of the applicants are Graduates. most of the applicants aren't self employed. most of the loans are above 360. most of the applicants do have a credit history.

##### Convert Loan_Status into Label Encoding #####
train.loc[:, 'Loan_Status'] = train.loc[:, 'Loan_Status'].map({'Y': 1, 'N': 0})

train.head()

train.columns

train.loc[:, 'Loan_Status'].value_counts()

for col in train.select_dtypes('float'):
    if col != 'Credit_History' and col != 'Loan_Amount_Term' :   # credit_history(1/0).
        plt.figure(figsize=[10,4])
        plt.title(col)
        sns.distplot(x = train[col])

num_features = train.select_dtypes('float').columns

fig,ax = plt.subplots(1,3,figsize=(20,5))

for i,feature in enumerate(num_features):
    sns.boxplot(data=train,x="Loan_Status",y=feature,ax=ax[i])
    ax[i].set_title(f'{feature} Box Plot')

#Handling outliers

train[train['CoapplicantIncome']<10000].shape
(586, 14)
sns.histplot(train[train['CoapplicantIncome']<10000]['CoapplicantIncome'], kde=True);

#we handle outliers by applying log transformation on LoanAmount and ApplicantIncome and handle them in CoapplicantIncome by setting a threshold.

df = train[train['CoapplicantIncome']<10000]

for col in ['LoanAmount','ApplicantIncome']:
    df[col] = df[col].apply(np.log1p)
fig,ax = plt.subplots(1,3,figsize=(20,5))

for i,feature in enumerate(num_features):
    sns.histplot(data=df,x=feature,ax=ax[i],kde=True,color='Green')
    ax[i].set_title(f'{feature} Distribution')

# Create subplots
fig = sp.make_subplots(rows=1, cols=2, subplot_titles=['Applicant Income Distribution', 'Applicant Income Box Plot'])

# Add histogram trace
histogram_trace = go.Histogram(x=train['ApplicantIncome'], nbinsx=30, marker=dict(color='rgba(0, 123, 255, 0.7)'))
fig.add_trace(histogram_trace, row=1, col=1)

# Add box plot trace
box_trace = go.Box(x=train['ApplicantIncome'], marker=dict(color='rgba(0, 123, 255, 0.7)'))
fig.add_trace(box_trace, row=1, col=2)

# Update layout
fig.update_layout(height=500, showlegend=False, title_text="Applicant Income Distribution and Box Plot")

# Display the plot
fig.show()

It can be deduced that the applicant income distribution is not normally distributed because the majority of the data point to the left. In later sections will attempt to normalise the data because normally distributed data is preferred by algorithms.

# Create subplots
fig = sp.make_subplots(rows=1, cols=2, subplot_titles=['Loan Amount Distribution', 'Loan Amount Box Plot'])

# Add histogram trace
histogram_trace = go.Histogram(x=train['LoanAmount'], nbinsx=30, marker=dict(color='rgba(0, 123, 255, 0.7)'))
fig.add_trace(histogram_trace, row=1, col=1)

# Add box plot trace
box_trace = go.Box(x=train['LoanAmount'], marker=dict(color='rgba(0, 123, 255, 0.7)'))
fig.add_trace(box_trace, row=1, col=2)

# Update layout
fig.update_layout(height=500, showlegend=False, title_text="Loan Amount Distribution and Box Plot")

# Display the plot
fig.show()

#We see a lot of outliers in this variable and the distribution is fairly normal. We will treat the outliers in later sections.

# Group by 'Loan_Status' and calculate mean 'ApplicantIncome'
income_mean = train.groupby('Loan_Status')['ApplicantIncome'].mean().reset_index()

# Create advanced bar chart using Plotly Express
fig = px.bar(income_mean, x='Loan_Status', y='ApplicantIncome',
             color='Loan_Status',
             labels={'ApplicantIncome': 'Mean Applicant Income'},
             title='Mean Applicant Income by Loan Status',
             text='ApplicantIncome',
             height=500)

# Update layout
fig.update_layout(xaxis_title='Loan Status', yaxis_title='Mean Applicant Income',
                  showlegend=False, barmode='group')

# Show the plot
fig.show()

#It can be inferred that Applicant’s income does not affect the chances of loan approval which contradicts our hypothesis in which we assumed that if the applicant’s income is high the chances of loan approval will also be high.

train['Total_Income'] = train['ApplicantIncome'] + train['CoapplicantIncome']
bins = [0, 2500, 4000, 6000, 81000]
group = ['Low', 'Average', 'High', 'Very high']
train['Total_Income_bin'] = pd.cut(train['Total_Income'], bins, labels=group)
Total_Income_bin = pd.crosstab(train['Total_Income_bin'], train['Loan_Status'])
Total_Income_bin.div(Total_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('Total_Income')
plt.ylabel('Percentage')
plt.show()

#We can see that Proportion of loans getting approved for applicants having low Total_Income is very less as compared to that of applicants with Average, High, and Very High Income.

bins=[0,100,200,700]
group=['Low','Average','High']
train['LoanAmount_bin']=pd.cut(train['LoanAmount'],bins,labels=group)
LoanAmount_bin=pd.crosstab(train['LoanAmount_bin'],train['Loan_Status'])
LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('LoanAmount')
P = plt.ylabel('Percentage')

#It can be seen that the proportion of approved loans is higher for Low and Average Loan Amounts as compared to that of High Loan Amounts which supports our hypothesis which considered that the chances of loan approval will be high when the loan amount is less.

# Plotting the histogram
sns.histplot(data=train, x="ApplicantIncome", kde=True, edgecolor="black", linewidth=1.2)

# Customizing the plot
plt.title("Distribution of Applicant Income", weight="bold")
plt.xlabel("Applicant Income")
plt.ylabel("Frequency")

# Display the plot
plt.show()

#Based on the provided histogram plot, it is evident that there are outliers in the data, as there are a few data points that deviate significantly from the majority of the values. Most of the income values range from 150 to 10000.

train["ApplicantIncome"].agg(["min", "max"])

train["CoapplicantIncome"].agg(["min", "max"])

train["Loan_Amount_Term"].agg(['min','max'])

plt.hist(x=train["CoapplicantIncome"], edgecolor="black", linewidth=1.2)

plt.title("Distribution of Coapplicant Income", weight="bold")
plt.xlabel("Coapplicant Income")
plt.ylabel("Frequency")

plt.show()

#Coapplicant income falls from 0 to 41000, but most of the income falls till 10000

train["LoanAmount"].agg(['min','max'])

# Plotting the histogram
sns.histplot(data=train, x="LoanAmount", kde=True, edgecolor="black", linewidth=1.2)

plt.title("Distribution of Loan Amount", weight="bold")
plt.xlabel("Loan Amount")
plt.ylabel("Frequency")

plt.show()

#Most of the Loan amt falls under 9 to 200

# Create the histogram plot
sns.histplot(data=train, x="Loan_Amount_Term", kde=True, edgecolor="black", linewidth=1.2)

# Customization options
plt.title("Loan Amount Term Distribution", weight="bold")  # Set the plot title
plt.xlabel("Loan Amount Term")  # Set the x-axis label
plt.ylabel("Frequency")  # Set the y-axis label

# Show the plot
plt.show()

#Most of the Loan term falls under 400

train['Credit_History'] = train['Credit_History'].map({1: "Yes", 0: "No"})

## Create countplot ==> Credit_History
sns.countplot(data=train, x="Credit_History")

## labels and title
plt.xlabel("Credit History")
plt.ylabel("Count")
plt.title("Distribution of Credit History", weight="bold")

## Show the plot
plt.show()

#Most of the applicants credit score is Yes (Good)

married_count = train["Married"].value_counts()
married_count

#train

#Most of the applicants credit score is Yes (Good)

gender_counts = train["Gender"].value_counts()
gender_counts

# Assuming train is your DataFrame
Gender_count = train["Gender"].value_counts()

# Create a histogram
sns.histplot(data=train, x="Gender", kde=True, edgecolor="black", linewidth=1.2)

# Labels and title
plt.xlabel("Gender")
plt.ylabel("Count")
plt.title("Distribution of Gender", weight="bold")

# Show the plot
plt.show()

#Male Applicants are more than Female Applicants

married_count = train["Married"].value_counts()
married_count

## Create a pie chart
plt.pie(x=married_count.values, labels=married_count.index, autopct="%0.2f%%")

## title
plt.title("Marital Status Distribution")

## Customize the styling
plt.axis("equal")  # Set the aspect ratio to make the pie circular

## Add a legend
plt.legend(loc='best')

## Show the plot
plt.show()

#Most of the Applicants are married

train["Dependents"].unique()

train["Dependents"] = train["Dependents"].map(
    {"Zero":"0","One":"1","Two":"2","Three plus":"3"}
     )
     
#train

train["Dependents"].value_counts(normalize=True)
Series([], Name: proportion, dtype: float64)
train["Education"].value_counts(normalize=True).to_frame()

train["Gender"].value_counts(normalize=True).to_frame()

train["Married"].value_counts(normalize=True).to_frame()

train["Self_Employed"].value_counts(normalize=True).to_frame()

#Only 14 % of the applicants are selfemployed

## Calculate the value counts of "Property_Area"
property_count = train["Property_Area"].value_counts()
print(property_count)
print("*"*50)

## Create a bar plot using Plotly
fig = px.bar(x=property_count.index, y=property_count.values, labels={"x": "Property Area", "y": "Count"},
                 title="<b>Distribution of Property Area", color_discrete_sequence=px.colors.qualitative.Bold_r)

## Customize the plot
fig.update_layout(
    paper_bgcolor="rgb(233,233,233)",
    plot_bgcolor="white",
)

## Show the plot
fig.show()

#Most of the applicants from semiurban

loan_count = train["Loan_Status"].value_counts()
print(loan_count)
print("*"*50)

## Create a pie chart
plt.pie(x=loan_count.values, labels=loan_count.index, autopct="%0.2f%%", shadow=True, explode=(0, 0.09))

## title
plt.title("Loan Status Distribution", weight="bold")

## Customize the styling
plt.axis("equal")  # Set the aspect ratio to make the pie circular

## Add a legend
plt.legend(loc='best')

## Show the plot
plt.show()

#The target feature in this dataset indicates the approval status of loans, distinguishing between acceptance and non-acceptance. It is noteworthy that there is an imbalance in the classes of this target variable, with a substantial disparity in the number of instances between accepted and non-accepted loan statuses.

#Is there a relationship between the applicant's income and loan amount requested?

## ApplicantIncome and LoanAmount
correlation_value =train["ApplicantIncome"].corr(train["LoanAmount"])
print(f'The correlation between Applicant Income and Loan Amount is {correlation_value:.2f}')

#There is a moderate relationship between income and loan amount.

## Create a scatter plot
sns.scatterplot(data=train, x="ApplicantIncome", y="LoanAmount")

## labels and title
plt.xlabel("Applicant Income")
plt.ylabel("Loan Amount")
plt.title("Relationship between Applicant Income and Loan Amount", weight="bold")

## style settings
plt.grid(True, linestyle="-")  # Add grid lines
plt.xlim(0, 80000)  # Set the x-axis limits
plt.ylim(0, 600)  # Set the y-axis limits

## Show the plot
plt.show()

##### The relationship of LoanAmount and ApplicantIncome #####
print('The relationship of LoanAmount and ApplicantIncome.\n')
sns.relplot(x = 'LoanAmount', y = 'ApplicantIncome', data = train)
plt.show()

print('*'*120,'\n')

##### The relationship of LoanAmount and ApplicantIncome with Gender #####
print('The relationship of LoanAmount and ApplicantIncome with Gender\n')
sns.relplot(x = 'LoanAmount', y = 'ApplicantIncome', hue = 'Gender', data = train)
plt.show()

print('*'*120,'\n')

##### The relationship of LoanAmount and ApplicantIncome with Gender, and self_employed #####
print('The relationship of LoanAmount and ApplicantIncome with Gender, and Self Employed\n')
sns.relplot(x = 'LoanAmount', y = 'ApplicantIncome', hue = 'Gender', col = 'Self_Employed', data = train)
plt.show()

#Loan amount depend on the applicants income

##### show the max and min loan #####
print('The maximum home loan amount is = {} '.format(train.loc[:, 'LoanAmount'].max(),'\n'))
print('The minimum home loan amount is = {} '.format(train.loc[:, 'LoanAmount'].min(),'\n'))
print('The average home loan amount is = {} '.format(train.loc[:, 'LoanAmount'].mean(),'\n'))

##### Show the distribution of Loan_Amount_Term #####
sns.distplot(train.loc[:, 'Loan_Amount_Term'])
plt.show()

print('\n')

sns.relplot(x = 'Loan_Amount_Term', y = 'LoanAmount', data = train)
plt.show()

##### Find out the sum, mean of Income and loan of creadit history #####
train.groupby(['Credit_History'])[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount','Loan_Amount_Term']].agg(['sum', 'mean'])

## Group the data by "Gender" and "Loan_Status" and calculate the count
loan_status_count = pd.crosstab(train['Gender'], train['Loan_Status'])

## bar plot
loan_status_count.plot(kind="bar", stacked=False)

## labels and title
plt.xlabel("Gender")
plt.ylabel("Count")
plt.title("Loan Status Count by Gender", weight="bold")

## Show the plot
plt.show()

train.groupby(["Education", "Loan_Status"])["Loan_Status"].count()

# Creating a count plot for "Loan_Status" with the hue based on "Education"
sns.countplot(data=train, x="Loan_Status", hue="Education")
plt.xlabel("Loan Status")   ## X-axis label
plt.ylabel("Count")          ## Y-axis label
plt.title("Loan Status by Education", weight="bold")  ## Plot title
plt.legend(title="Education", loc="upper right") ## legend
plt.show()

#Graduate's loan approvals are accepted more

print(train.groupby(["Loan_Status", "Credit_History"])["Loan_Status"].count())
print("*"*50)
sns.countplot(data=train, x="Loan_Status", hue="Credit_History")
plt.title("Loan Status by Credit History.", weight="bold")
plt.show()

#Mostly, anyone with a credit history of 1 (Yes) will be accepted

sns.countplot(data=train,x='Loan_Status');

loan_status_married =train.groupby(["Loan_Status", "Married"])["Married"].count().unstack()
loan_status_married

print("the ratio of Non-married rejected".title(),
      round((loan_status_married.iloc[0,0]) / (loan_status_married.iloc[0,:].sum(axis=0)),2)*100)
print("the ratio of married rejected".title(),
      round((loan_status_married.iloc[1,0]) / (loan_status_married.iloc[1,:].sum(axis=0)),2)*100)

train.groupby("Property_Area")["LoanAmount"].agg(["sum","mean","count"]).sort_values(by="mean", ascending=False)

# Create the bar plot
sns.barplot(data=train, x="Property_Area", y="LoanAmount", errorbar=None, estimator="mean")

# Customize the plot
plt.title("Mean LoanAmount by Property Area", weight="bold")
plt.xlabel("Property Area")
plt.ylabel("Mean LoanAmount")
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# Show the plot
plt.show()

#The average Loan Amount in the "Rural" area is higher than in the other areas.

#Education column
sns.countplot(train['Education'],palette='RdBu')

#Self employed column
train['Self_Employed'].value_counts().plot(kind = 'pie',autopct = '%.2f',colors=['pink','violet'])

#Property area column
train['Property_Area'].value_counts().plot(kind = 'pie',autopct = '%.2f',colors=['violet','pink','blue'])

#As indicated by the box plot there are many outliers that can deteriorate the model. Thus they need to be handled.

train.head()

#But first let us have a look at correlations between columns

train1=train.drop(['Loan_ID','Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area','Total_Income_bin','LoanAmount_bin'],axis='columns')
train1.corr()

train[train['ApplicantIncome']<train['CoapplicantIncome']]['Loan_Status'].value_counts()

## Create the countplot
countplot = sns.countplot(data=train, x="Self_Employed", hue="Loan_Status")  # Customize the color palette

## Add labels and a title
countplot.set(xlabel="Self Employed", ylabel="Count")
plt.title("Count of Loan Status by Self Employment", weight="bold")

## set legend
legend = countplot.get_legend()
legend.set_title("Loan Status")
for t, l in zip(legend.texts, ["Approved", "Not Approved"]):
    t.set_text(l)

## axis labels and tick labels
countplot.set_xticklabels(countplot.get_xticklabels(), rotation=45)  # Rotate x-axis labels

## show plot
plt.show()

sns.pairplot(data=train, hue="Loan_Status")
plt.show()

# Calculate the correlation between 'LoanAmount' and 'Loan_Amount_Term'
correlation = train[["LoanAmount", "Loan_Amount_Term"]].corr().iloc[0, 1]

# Print the correlation value
print("Correlation between Loan Amount and Loan Amount Term = {:.2f}".format(correlation))

# Create a scatterplot
sns.scatterplot(data=train, x="LoanAmount", y="Loan_Amount_Term")

# Set plot labels and title
plt.xlabel("Loan Amount")
plt.ylabel("Loan Amount Term")
plt.title("Scatterplot of Loan Amount vs Loan Amount Term", weight="bold")

# Show the plot
plt.show()

#Correlation between Numerical columns
Numerical_cols = train.select_dtypes(include="number").columns.to_list() ## Numerical Features in the data

## Compute the correlation matrix
correlation_matrix = train.corr(numeric_only=True)

## Create the heatmap
sns.heatmap(correlation_matrix, annot=True, cmap="RdBu_r", center=0)

plt.show()

#Correlation between Loan Amount and ApplicantIncome by Loan Status

## Create Scatterplot
scatterplot = sns.scatterplot(data=train, x="ApplicantIncome", y="LoanAmount", hue="Loan_Status")

## title and labels
plt.title("Applicant Income vs. Loan Amount", weight="bold")
plt.xlabel("Applicant Income")
plt.ylabel("Loan Amount")

## legend settings
scatterplot.legend(title="Loan Status")
legend_labels = ["Approved", "Not Approved"]  # Custom legend labels
for t, l in zip(scatterplot.get_legend().texts, legend_labels):
    t.set_text(l)

## show the plot
plt.show()

#Feature Engineering

data = train.copy()

## Total Income
data["TotalIncome"] = data["ApplicantIncome"] + data["CoapplicantIncome"]

## Create KDE plot for total income
sns.kdeplot(data=data, x="TotalIncome", fill=True)
plt.title("Total Income Distribution before log transform".title())
plt.show()

#EMI, which stands for Equated Monthly Installment, is a fixed payment made by a borrower to a lender at a specified date each month.

## EMI
data["EMI"] = data["LoanAmount"] / data["Loan_Amount_Term"]
data.columns.to_list()

train.replace({'Yes':1,'No':0},inplace = True)
train.head()

train.replace({'Graduate':1,'Not Graduate':0},inplace = True)
train.head()

train.replace({'Urban':1,'Rural':0,'Semiurban':2},inplace = True)
train.head()

train.replace({'Y':1,'N':0},inplace = True)
train.head()

#Applicant income column
train[train['ApplicantIncome']>50000]

train= train[train['ApplicantIncome']<50000]
train.shape

#Coapplicant income
train[train['CoapplicantIncome']>=20000]

train = train[train['CoapplicantIncome']<20000]
train.shape

#Loan amount column
train[train['LoanAmount']>=600]

#train

train['Dependents'].unique()
array([nan])
train=train[train['LoanAmount']<600]
train.shape

train2 = train[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Status']].copy()
train2.head()

##### Create a feature named annual income #####
train['annual_income'] = train.loc[:, 'ApplicantIncome']*12

train.head()

##### What's the ratio of annual income based on Education and Employement #####
def find_query(group):
  return group['annual_income'].value_counts()

train.groupby([ 'Education', 'Self_Employed']).apply(find_query).head()

##### Find the relationship of CoapplicantIncome    and ApplicantIncome #####
print('The relationship of CoapplicantIncome    and ApplicantIncome is given below.\n')
sns.relplot(x = 'CoapplicantIncome', y = 'ApplicantIncome', data = train)
plt.show()

print('*'*120,'\n')

##### Find the relationship of CoapplicantIncome    and ApplicantIncome with Dependents #####
print('The relationship of CoapplicantIncome    and ApplicantIncome with Dependents is given below.\n')
sns.relplot(x = 'CoapplicantIncome', y = 'ApplicantIncome', hue = 'Dependents_', data = train)
plt.show()

sns.pairplot(train2,hue = 'Loan_Status')

train1.corr()['Loan_Status']

#Now, moving on to handling text data

#As a machine learning model only understands numeric values, thus the text data or columns should be converted to some numeric equivalent. This is the reason all the yes and no were changed to 1, 0 and the column gender was also changed to numeric values.

train.info()

# Select numerical columns
numerical_columns = train.select_dtypes(include=['number'])

# Calculate the correlation matrix
matrix = numerical_columns.corr()

# Create a heatmap
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(matrix, vmax=0.8, square=True, cmap="BuPu")

# Show the plot
plt.show()

#We see that the most correlated variables are (ApplicantIncome – LoanAmount) and (Credit_History – Loan_Status). LoanAmount is also correlated with CoapplicantIncome.

train.head(1)

test.shape

train['Credit_History'] = train['Credit_History'].astype(int)
train['Credit_History'].dtype
train.head()

train=train.drop(['Loan_ID','Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area','Total_Income_bin','LoanAmount_bin'],axis='columns')

#Data Splitting and Preprocessing

from sklearn.metrics import confusion_matrix
def prediction_report(model,X_test,y_test,color):

    #'''this function is to evaluate the model:
    1--> print the classification report     2--> display the confusion matrix'''

    #test report
    y_pred_test = model.predict(X_test)
    print(classification_report(y_pred_test,y_test))
    
    #test confusion matrix
    plt.figure(figsize=(4,3))
    sns.heatmap(confusion_matrix(y_test,y_pred_test), cmap=color, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix');
    
from sklearn.model_selection import GridSearchCV
def model_tunning(model,X_train,y_train,params):

    #'''This function recieves a model then tune it using GridSearch
    then print the best parameters and return the best estimator'''

    grid_search = GridSearchCV(model, param_grid=params, cv = 5, scoring='f1')
    grid_search.fit(X_train,y_train)
    print(grid_search.best_params_)
    print('Mean cross-validated f1 score of the best estimator is: ',grid_search.best_score_)
    return grid_search.best_estimator_
    
from sklearn.model_selection import train_test_split
features = train.columns.drop(['Loan_Status','Loan_Amount_Term'])
target = 'Loan_Status'

X = train[features]
y = train[target]

num_features = X.select_dtypes('number').columns
cat_features = X.select_dtypes('object').columns

X_train , X_test , y_train , y_test = train_test_split(X , y , random_state=42 , test_size=0.2 , stratify = y)

# import libraries
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.compose import make_column_transformer
from sklearn.impute import KNNImputer

#numerical features pipeline
num_pipeline = make_pipeline(
    KNNImputer(),
    StandardScaler(),
)
#categorical features pipeline
cat_pipeline = make_pipeline(
    SimpleImputer(strategy = 'most_frequent'),
    OneHotEncoder(),
)
#combine both pipelines
preprocessor = make_column_transformer(
    (num_pipeline,num_features),
    (cat_pipeline,cat_features)
)

#Logistic Regression

log_reg = make_pipeline(
    preprocessor,
    LogisticRegression(random_state=42, solver='liblinear', max_iter = 5000)
)

param_grid = {
    'logisticregression__penalty':['l1','l2'],
    'logisticregression__C':[0.001, 0.01, 0.1, 1, 10, 100, 1000]
}

log_reg = model_tunning(log_reg,X_train,y_train,param_grid)

#training report
from sklearn import metrics
from sklearn.metrics import classification_report
prediction_report(log_reg,X_train,y_train,'Blues')

#test report
prediction_report(log_reg,X_test,y_test,'Blues')

#Support Vector Classifier (SVC)

from sklearn.svm import SVC
svm = make_pipeline(
    preprocessor,
    SVC(kernel = 'poly',random_state=42)
)

param_grid = {
    'svc__C':[ 0.01, 0.1, 1, 10, 100],
    'svc__degree': np.arange(2,5),
}

svm = model_tunning(svm,X_train,y_train,param_grid)

#training report
prediction_report(svm,X_train,y_train,'Greens')

prediction_report(svm,X_test,y_test,'Greens')

#KNN Classifier

from sklearn.neighbors import KNeighborsClassifier
knn = make_pipeline(
    preprocessor,
    KNeighborsClassifier()
)

param_grid={
        'kneighborsclassifier__n_neighbors':range(1,21,2),
        'kneighborsclassifier__weights':['uniform','distance'],
        'kneighborsclassifier__metric':['euclidean','manhattan']
}

knn = model_tunning(knn,X_train,y_train,param_grid)

#training report
prediction_report(knn,X_train,y_train,'Reds')

#testing report
prediction_report(knn,X_test,y_test,'Reds')

#Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier
dec_tree = make_pipeline(
    preprocessor,
    DecisionTreeClassifier()
)
param_grid = {
    'decisiontreeclassifier__max_depth': np.arange(2, 15),
    'decisiontreeclassifier__min_samples_split': np.arange(2, 7),
    'decisiontreeclassifier__min_samples_leaf': np.arange(1, 6),
}

dec_tree = model_tunning(dec_tree,X_train,y_train,param_grid)

prediction_report(dec_tree,X_train,y_train,'Blues')

prediction_report(dec_tree,X_test,y_test,'Blues')

#Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
rfc = make_pipeline(
    preprocessor,
    RandomForestClassifier(n_jobs=-1)
)
param_grid = {
    'randomforestclassifier__max_depth': np.arange(2, 8),
    'randomforestclassifier__n_estimators': np.arange(10, 101, 10),
}
rfc = model_tunning(rfc,X_train,y_train,param_grid)

prediction_report(rfc,X_train,y_train,'Blues')

prediction_report(rfc,X_test,y_test,'Blues')

#Conclusion The best performing model is random forest classifier with f1 score of 81% on training set and 83 % score on test set.

from sklearn.model_selection import train_test_splitX_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=10)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import graphviz

#Now that we know random forest classifier works better on the data, lets train the model.

X_train.columns

y_train.unique()

def select_best_model(x,y):
    algos = {
        'KNN' : {
            'model' : KNeighborsClassifier(),
            'params' : {
                'n_neighbors' : [13,15],
                'weights' : ['uniform', 'distance']
            }
        },
        'Random_forest_classifier' : {
            'model' : RandomForestClassifier(),
            'params' : {
                'n_estimators' : [150,250],
                'bootstrap' : [True,False]
            }
        }
    }
    scores=[]
    cv = ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
    for algos_name, config in algos.items():
        gs = GridSearchCV(config['model'],config['params'],cv=cv)
        gs.fit(x,y)
        scores.append({
            'model' : algos_name,
              "best_params" :gs.best_params_,
            "best_score" : gs.best_score_
        })
    return pd.DataFrame(scores, columns=['model', 'best_params', 'best_score'])
select_best_model(X,y)

#Now that we know random forest classifier works better on the data, lets train the model.

rf_clf = RandomForestClassifier(n_estimators=250)
rf_clf.fit(X_train,y_train)

y_pred = rf_clf.predict(X_test)
y_pred

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True)

from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,y_pred)

rf_clf.score(X_test,y_test)

With this the Notebook comes to its conclusion, following are the conclusions:

#The data had many null values. Thus, one by one all the null values were filled. There were some outliers in the data which was also handled. The model performed the best on random forest among the two tested. Giving the model an error of approx. 24% with some wrong predictions but all in all the model performed fine. Now, the accuracy can be increased or the error can be decreased by testing the data on some other algorithm and by doing some more hyperparameter tuning. But for now this is it.

Using K-Means Clustering in this test dataframe

##### Preprocessing the test data frame #####
from sklearn.preprocessing import MinMaxScaler
MinMaxScaler_ = StandardScaler()
test_scaled = pd.DataFrame(MinMaxScaler_.fit_transform(X_test), columns = X_test.columns)
test_scaled.head()

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
wcss_ = []
for n_cluster in range(2, 5):
  KMeans_ = KMeans(n_clusters = n_cluster, max_iter = 500)
  KMeans_.fit_predict(X_test)
  wcss_.append(KMeans_.inertia_)
  print(f'The score is = {silhouette_score(X_test.values, KMeans_.labels_)} for n_cluster = {n_cluster}')

##### Take n_cluster = 2 in KMeans #####
KMeans_ = KMeans(n_clusters = 2, max_iter = 500)
target_ = KMeans_.fit_predict(X_test)
print(target_)

df1_ = pd.concat([X_test, pd.DataFrame(y, columns = ['Loan_Status'])], axis = 1, ignore_index = True)
df1_.head()

pd.DataFrame(MinMaxScaler_.fit_transform(pd.get_dummies(data = X_train, drop_first = True)))

new_df = pd.concat([X_train, X_test], axis = 0, ignore_index = True)
new_df.head()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(LogisticRegression(),{'C':[0.001,0.01,0.1,1,10]}, cv=10).fit(X_train, y_train)
print("Best Params " + str(clf.best_params_) + " Best Score " + str(clf.best_score_))

#building model 1 with random forest classifier and predict labels
model1=RandomForestClassifier(n_estimators=600,max_depth=10)
model1.fit(X_train,y_train)
tr_pred=model1.predict(X_train)
print(classification_report(y_train,tr_pred))

#predictions with X_val data set
ts_pred=model1.predict(X_test)
print(classification_report(y_test,ts_pred))

#model 2 with HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
model2=HistGradientBoostingClassifier(max_iter=800,learning_rate=0.01)
model2.fit(X_train,y_train)
ts_pred=model2.predict(X_test)
print(classification_report(y_test,ts_pred))

#model 3 with logisticRegression
model3=LogisticRegression(max_iter=2000)
model3.fit(X_train,y_train)
ts_pred=model3.predict(X_test)
print(classification_report(y_test,ts_pred))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(LogisticRegression(),{'C':[0.001,0.01,0.1,1,10]}, cv=10).fit(X_train, y_train)
print("Best Params " + str(clf.best_params_) + " Best Score " + str(clf.best_score_))

from sklearn.model_selection import cross_val_predict
yp = cross_val_predict(clf, X_train, y_train, cv=10)
from sklearn.metrics import classification_report
print(classification_report(y_train, yp))

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
ConfusionMatrixDisplay.from_predictions(y_train, yp, cmap="Blues")
plt.show()

ypt = clf.predict(X_test)
print(classification_report(y_test, ypt))

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
ConfusionMatrixDisplay.from_predictions(y_test, ypt, cmap="Blues")
plt.show()

Feature Selection Wrapper (Forward)

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.feature_selection import SequentialFeatureSelector
sw = SequentialFeatureSelector(SVC(),n_features_to_select=5,direction='forward', cv=5).fit(X_train, y_train)
X_train_norm_sw = sw.transform(X_train)
X_test_norm_sw = sw.transform(X_test)
clf = GridSearchCV(SVC(),{'C':[1,2,4,8,16,32]}, cv=10).fit(X_train_norm_sw, y_train)

yp = cross_val_predict(clf.best_estimator_, X_train_norm_sw, y_train, cv=10)
print(classification_report(y_train, yp))
ConfusionMatrixDisplay.from_predictions(y_train, yp, cmap="Blues")
plt.show()

ypt = clf.predict(X_test_norm_sw)
print(classification_report(y_test, ypt))
ConfusionMatrixDisplay.from_predictions(y_test, ypt, cmap="Blues")
plt.show()
