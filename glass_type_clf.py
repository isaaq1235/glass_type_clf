# S2.1: Open Sublime text editor, create a new Python file, copy the following code in it and save it as 'glass_type_app.py'.
# You have already created this ML model in ones of the previous classes.

# Importing the necessary Python modules.
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

# ML classifier Python modules
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data() 

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# S3.1: Create a function that accepts an ML model object say 'model' and the nine features as inputs 
# and returns the glass type.

@st.cache()
def prediction(model,RI,Na,Mg,Al,Si,K,Ca,Ba,Fe):
  gl_t = model.predict([[RI,Na,Mg,Al,Si,K,Ca,Ba,Fe]])

  if gl_t[0] == 1:
    return "building windows float processed"
  
  elif gl_t[0] == 2:
    return "building windows non float processed"

  elif gl_t[0] == 3:
    return "vehicle windows float processed"

  elif gl_t[0] == 4:
    return "vehicle windows non float processed"
  
  elif gl_t[0] == 5:
    return "containers"

  elif gl_t[0] == 6:
    return "tableware"

  else:
    return "headlamp"
  
# S4.1: Add title on the main page and in the sidebar.

st.title("Glass Type Predictor")

st.sidebar.title("Exploratory Data Analysis")

# S5.1: Using the 'if' statement, display raw data on the click of the checkbox.

if st.sidebar.checkbox("Show raw data"):
  st.subheader("Raw Dataset")
  st.dataframe(glass_df)

# S6.1: Scatter Plot between the features and the target variable.
# Add a subheader in the sidebar with label "Scatter Plot".

st.sidebar.subheader("Plots")

# Choosing x-axis values for the scatter plot.
# Add a multiselect in the sidebar with the 'Select the x-axis values:' label

features_list = st.sidebar.multiselect('Select the x-axis values', ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))

# and pass all the 9 features as a tuple i.e. ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe') as options.

# S6.2: Create scatter plots between the features and the target variable.
# Remove deprecation warning.
st.set_option('deprecation.showPyplotGlobalUse', False)



# S6.3: Create histograms for all the features.
# Sidebar for histograms.



# glass_type_count = glass_df.groupby(by = "GlassType")

# fig = px.scatter(glass_df, x = np.linspace(glass_df["RI"].min(),glass_df["RI"].max()+1, glass_df.shape[0]), y = glass_df["RI"], size = "GlassType", color = "GlassType", color_continuous_scale = px.colors.sequential.Viridis, title = "Scatter Plot of RI against Glass Type")

# fig.show()

# fig.write_html('barplot.html')

# S1.1: Remove the multiselect widgets for histograms and box plots and add a new multiselect widget to choose a type of visualisation.
# Sidebar subheader for scatter plot

# Remove deprecation warning.
st.set_option('deprecation.showPyplotGlobalUse', False)

# Choosing x-axis values for scatter plots.

# Creating scatter plots.

# Remove the code blocks for histogram and box plots.

# Add a subheader in the sidebar with label "Visualisation Selector"



st.sidebar.subheader("Visualistaion Selector")
# Add a multiselect in the sidebar with label 'Select the Charts/Plots:'
plot_types = st.sidebar.multiselect('Select the Charts/Plots:', ('Scatter Plot','Histogram', 'Box Plot', 'Count Plot', 'Pie Chart', 'Correlation Heatmap', 'Pair Plot'))
# and with 6 options passed as a tuple ('Histogram', 'Box Plot', 'Count Plot', 'Pie Chart', 'Correlation Heatmap', 'Pair Plot').
# Store the current value of this widget in a variable 'plot_types'.


if 'Scatter Plot' in plot_types:
    for i in feat_list:
        st.subheader(f"Scatter Plot between {i} and Glass Type")
        plt.figure(figsize = (16,9))
        sns.scatterplot(x = i, y = "GlassType", data = glass_df)
        st.pyplot()

if 'Histogram' in plot_types:
  for i in feat_list:
    st.subheader(f"Histogram between {i} and Glass Type")
    plt.figure(figsize = (16,9))
    plt.hist(x = glass_df[i])
    st.pyplot()

if 'Box Plot' in plot_types:
  # plot box plot
  for i in feat_list:
    st.subheader(f"Boxplot between {i} and Glass Type")
    plt.figure(figsize = (16,9))
    sns.boxplot(x = glass_df[i])
    st.pyplot()

if 'Count Plot' in plot_types:
  # plot count plot 
    st.subheader(f"Countplot of the Glass Types")
    plt.figure(figsize = (16,9))
    sns.countplot(x = glass_df["GlassType"].value_counts())
    st.pyplot()

if 'Pie Chart' in plot_types:
  # plot pie chart
    st.subheader(f"Pie Chart of Glass Types")
    plt.figure(figsize = (16,9))
    pie_data = glass_df["GlassType"].value_counts()
    plt.pie(pie_data, labels = pie_data.index, autopct = '%1.2f%%', startangle = 30, explode = np.linspace(.06, .16, 6))
    st.pyplot()
if 'Correlation Heatmap' in plot_types:
  # plot correlation heatmap
    st.subheader(f"Heatmap of Glass Types")
    plt.figure(figsize = (16,9))
    ax = sns.heatmap(glass_df.corr(), annot = True)
    bottom,top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    st.pyplot()

if 'Pair Plot' in plot_types:
  # plot pair plot
    st.subheader(f"Heatmap of Glass Types")
    plt.figure(figsize = (16,9))
    sns.pairplot(glass_df)
    st.pyplot()


st.sidebar.subheader("Select the features")

feat_list2 = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']

RI = st.sidebar.slider(feat_list2[0],float(glass_df['RI'].min()),float(glass_df['RI'].max()))

Na = st.sidebar.slider(feat_list2[1],float(glass_df['Na'].min()),float(glass_df['Na'].max()))

Mg = st.sidebar.slider(feat_list2[2],float(glass_df['Mg'].min()),float(glass_df['Mg'].max()))

Al = st.sidebar.slider(feat_list2[3],float(glass_df['Al'].min()),float(glass_df['Al'].max()))

Si = st.sidebar.slider(feat_list2[4],float(glass_df['Si'].min()),float(glass_df['Si'].max()))

K = st.sidebar.slider(feat_list2[5],float(glass_df['K'].min()),float(glass_df['K'].max()))

Ca = st.sidebar.slider(feat_list2[6],float(glass_df['Ca'].min()),float(glass_df['Ca'].max()))

Ba = st.sidebar.slider(feat_list2[7],float(glass_df['Ba'].min()),float(glass_df['Ba'].max()))

Fe = st.sidebar.slider(feat_list2[8],float(glass_df['Fe'].min()),float(glass_df['Fe'].max()))


classifier = st.sidebar.selectbox('Classifier', ('Support Vector Machine', 'Logistic Regression', 'Random Forest Classifier'))

if classifier == 'Support Vector Machine':
    st.sidebar.subheader("Model Hyperparameters")
    c_val = st.sidebar.number_input("C: ",1,50,2)
    gamma_val = st.sidebar.number_input("Gamma: ",1,50,2)
    kernel_val = st.sidebar.selectbox('Kernel:', ("linear", "rbf", "poly"))


    if st.sidebar.button("Predict"):
        st.subheader("Support Vector Machine Prediction:\n")

        svc_model = SVC(kernel = kernel_val,C = c_val,gamma = gamma_val)

        svc_model.fit(X_train,y_train)

        y_pred = svc_model.predict(X_test)


        glass_pred = prediction(svc_model, RI, Na, Mg, Al, Si, K, Ca, Ba, Fe)

        st.write("Glass Type = ", glass_pred)


        plot_confusion_matrix(svc_model,X_test,y_test)
        st.pyplot()

        st.write(f"The accuracy score of this model is: {svc_model.score(X_train,y_train):.2f}")

if classifier == 'Random Forest Classifier':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators_input = st.sidebar.number_input("Number of trees in the forest", 100, 5000, step = 10)
        max_depth_input = st.sidebar.number_input("Maximum depth of the tree", 1, 100, step = 1)

    # If the user clicks 'Classify' button, perform prediction and display accuracy score and confusion matrix.
    # This 'if' statement must be inside the above 'if' statement. 
        if st.sidebar.button('Classify'):
            st.subheader("Random Forest Classifier")
            rf_clf = RandomForestClassifier(n_estimators = n_estimators_input, max_depth = max_depth_input, n_jobs = -1)
            rf_clf.fit(X_train,y_train)
            accuracy = rf_clf.score(X_test, y_test)
            glass_type = prediction(rf_clf, RI, Na, Mg, Al, Si, K, Ca, Ba, Fe)
            st.write("The Type of glass predicted is:", glass_type)
            st.write("Accuracy", accuracy.round(2))
            plot_confusion_matrix(rf_clf, X_test, y_test)
            st.pyplot()

if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        c_val = st.sidebar.number_input("C:", 1, 100, step = 2)
        max_iter_val = st.sidebar.number_input("Maximum Iterations", 10, 1000, step = 10)

    # If the user clicks 'Classify' button, perform prediction and display accuracy score and confusion matrix.
    # This 'if' statement must be inside the above 'if' statement. 
        if st.sidebar.button('Classify'):
            st.subheader("Logistic Regression")
            log_reg = LogisticRegression(C = c_val, max_iter = max_iter_val)
            log_reg.fit(X_train,y_train)
            accuracy = log_reg.score(X_test, y_test)
            glass_type = prediction(log_reg, RI, Na, Mg, Al, Si, K, Ca, Ba, Fe)
            st.write("The Type of glass predicted is:", glass_type)
            st.write("Accuracy", accuracy.round(2))
            plot_confusion_matrix(log_reg, X_test, y_test)
            st.pyplot()



