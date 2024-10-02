import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import graphviz
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from general import Logger
import io
from sklearn.preprocessing import minmax_scale, LabelEncoder
import warnings
from math import ceil
import time

# Target variable name is assigned here and used in the code globally
target_col = "y"

def load_data(file_path : str, logger : Logger = None) -> pd.DataFrame:
    # opening the file and show 5 first rows
    logger.log("#################### First 5 rows of original data #######################")
    # open csv file and load it into a dataframe
    data = pd.read_csv(file_path)
    # return the first 5 rows of the dataset
    logger.log(data.head())
    return data

def describe_data(data : pd.DataFrame, logger : Logger = None, suffix : str = "original"):
    # extract some descriptive analysis
    logger.log(f"\n#################### Describe {suffix} data specifications ###############")

    # show the rows and columns count plus columns data types
    info = io.StringIO()
    data.info(buf=info)
    logger.log(info.getvalue())

    # show statistic of the columns one by one
    describe = data.describe()

    # since median and mode are not included in describe method outputs, they are added manually to the output for numeric columns
    median = data.median(axis="index", numeric_only=True)
    numeric_mode = data.mode(numeric_only=True)
    for col in describe:
        describe.at["median",col] = median[col]
        describe.at["mode",col] = numeric_mode.loc[0,col]

    logger.log(describe)
    
    # for categorical columns only mode measure is available
    categorical_mode = data.mode().drop(columns=describe.columns)
    logger.log(f"\nModes of categorical variables are:\n{categorical_mode}")
    for col in categorical_mode:
        logger.log(f"\n{data.value_counts([col])}")

    logger.log("##########################################################################\n")


def handle_missing_values(data : pd.DataFrame, logger : Logger = None) -> pd.DataFrame:
    # examining data to discover missing values
    logger.log(f"Dataset has these missing values:\n{data.isna().sum()}\n")
    # Replace missing values via bfill method
    data = data.bfill()
    logger.log(f"Missing values are replaced by next valid value of that column using bfill method.\n")
    return data

def handle_duplicate_values(data : pd.DataFrame, logger : Logger = None) -> pd.DataFrame:
    # explorirng data to find duplicated rows
    logger.log(f"Duplicate rows are:\n{data.loc[data.duplicated()]}")
    logger.log(f"Duplicate rows count: {data.duplicated().sum()}")
    # eliminate duplicate rows from data
    data = data.drop_duplicates()
    # summary about data after removing duplicates
    logger.log(f"After removing duplicate rows the dataset has {data.shape[0]} rows.\n")
    return data

def handle_outlier_values(data : pd.DataFrame, output_folder : str = "output", logger : Logger = None) -> pd.DataFrame:
    # Distinguish and remove the outliers based on IQR analysis
    describe = data.describe()
    try:
        describe = describe.drop(columns=target_col)
    except:
        pass
    # Box plot to show outliers
    plot_box(data=data, output_folder=output_folder, logger=logger)

    # Detecting outliers with IQR method
    logger.log("Outliers based on IQR method and skewness before handling them:\n")

    outliers_index = {}
    for col in describe.columns:
        # acquiring q1 and q3 to establish allowed area
        q1 = describe.loc["25%", col]
        q3 = describe.loc["75%", col]
        iqr = q3 - q1
        # every value outside of allowed area is distiguished as an outlier
        outliers = data[col].loc[(data[col] > q3 + 1.5*iqr) | (data[col] < q1 - 1.5*iqr)]
        logger.log(f"Outliers of Column {col} count: {outliers.count()}  ---  skewness: {data[col].skew()}")
        # outlier index of each column
        if not outliers.empty:
            outliers_index[col] = outliers.index

    # data = data.drop(outliers_index)
    # # summary about data after removing outliers
    # logger.log(f"After removing outliers rows the dataset has {data.shape[0]} rows.\n")

    # Replace outliers with mode of that columns
    for col in outliers_index.keys():
        data.loc[outliers_index[col],col] = data.mode(numeric_only=True).loc[0,col]
    
    # Detecting outliers with IQR method after handling them
    logger.log("\nOutliers based on IQR method and skewness after handling them:\n")
    
    outliers_index = {}
    for col in describe.columns:
        # acquiring q1 and q3 to establish allowed area
        q1 = describe.loc["25%", col]
        q3 = describe.loc["75%", col]
        iqr = q3 - q1
        # every value outside of allowed area is distiguished as an outlier
        outliers = data[col].loc[(data[col] > q3 + 1.5*iqr) | (data[col] < q1 - 1.5*iqr)]
        logger.log(f"Outliers of Column {col} count: {outliers.count()}  ---  skewness: {data[col].skew()}")
        # outlier index of each column
        if not outliers.empty:
            outliers_index[col] = outliers.index

    return data


def encode_data(data : pd.DataFrame, logger : Logger) -> pd.DataFrame:
    # Encode categorical variables to numeric
    # Replace encoded variable using Label Encoding

    logger.log(f"\nBelow columns need encoding:")

    le = LabelEncoder()
    # for each columns in the dataset
    for col in data.columns:
        # if the column is categorical
        if data[col].dtype == "object":
            # encode the data into numeric
            data[col] = le.fit_transform(data[col])
            #unique values of encoded column
            logger.log(f"Values for {col} are {le.classes_}")

    return data


def scale_data(data : pd.DataFrame, logger : Logger = None) -> pd.DataFrame:
    # Scale features values

    # Scale all columns except target column
    scaling_cols = list(data.drop(target_col,axis="columns").columns)

    # Extract columns which need Scaling
    need_scaling = data[scaling_cols]

    # Scale data
    ndarr = minmax_scale(need_scaling, axis=0) #axis=0 means column-wise

    scaled_data = pd.DataFrame()
    # Build dataframe based on the ndarray and original data
    scaled_data.index = data.index
    for col in data.columns:
        # all features values come from ndarr and the target values come from original data
        if col in scaling_cols:
            scaled_data[col] = ndarr[:,scaling_cols.index(col)]
        else:
            scaled_data[col] = data.loc[:,col]

    # summary about data after encoding and scaling
    logger.log(f"\nAfter encoding and Scaling dataset is:")
    logger.log(scaled_data)
    logger.log(f"'No' Count: {scaled_data[target_col].count() - scaled_data[target_col].sum()} --- 'Yes' Count: {scaled_data[target_col].sum()}\n")

    return scaled_data

def plot_correlation(data : pd.DataFrame, suffix : str = "", output_folder : str = "output", logger : Logger = None):
    # Extract correlatoin between features themselves and also with label plus plot their heatmap
    # set the figure resolution and dpi
    fig = plt.figure(figsize=(16, 9), dpi=600)
    correlations = data.corr()
    logger.log(f"correlations of features:\n{correlations}")
    sns.heatmap(correlations,annot=True)
    # Save the file with proper dpi
    plt.savefig(fname=f"{output_folder}/correlations{('_' if suffix else '') + suffix}.png", format="png", dpi = fig.dpi)
    logger.log(f"##################### {output_folder}/correlations{('_' if suffix else '') + suffix}.png file saved ###################\n")
   

def plot_pair(data : pd.DataFrame, suffix : str = "", output_folder : str = "output", logger : Logger = None):
    # show data exploration by plotting relationship between every two columns
    # set the figure resolution and dpi
    fig = plt.figure(figsize=(16, 9), dpi=600)
    # data Exploration
    sns.pairplot(data)
    # Save the file with proper dpi
    plt.savefig(fname=f"{output_folder}/data_exploration{('_' if suffix else '') + suffix}.png", format="png", dpi = fig.dpi)
    logger.log(f"##################### {output_folder}/data_exploration{('_' if suffix else '') + suffix}.png file saved ###################\n")


def plot_box(data : pd.DataFrame, suffix : str = "", output_folder : str = "output", logger : Logger = None):
    # Boxplots of input of dataset (features) to show outliers
    # set the figure resolution and dpi
    fig = plt.figure(figsize=(16, 9), dpi=600)

    # Distinguish and remove the outliers based on IQR analysis
    describe = data.describe()
    row_count = 1
    col_count = len(describe.columns)

    i = 1
    for col in describe.columns:
        # Creating inputs subplots
        plt.subplot(row_count, col_count, i)
        sns.boxplot(data[col])
        plt.title(f"{col} (input)")
        i += 1

    # setup the layout to fit in the figure
    plt.tight_layout(pad=1, h_pad=0.5, w_pad=0.5) 
    # Save the file with proper dpi
    plt.savefig(fname=f"{output_folder}/input_outliers{('_' if suffix else '') + suffix}.png", format="png", dpi = fig.dpi)
    logger.log(f"##################### input_outliers{('_' if suffix else '') + suffix}.png file saved ######################\n")


def plot_hist(data : pd.DataFrame, suffix : str = "", output_folder : str = "output", logger : Logger = None):
    # histogram of input and output of dataset (features and label)
    # set the figure resolution and dpi
    fig = plt.figure(figsize=(16, 9), dpi=600)

    row_count = ceil((data.shape[1]-1)/3)
    col_count = 4

    i = 1
    for col in data.drop(columns=target_col).columns:
        # Creating inputs subplots
        plt.subplot(row_count, col_count, i)
        sns.histplot(data[col], kde=True)
        plt.title(f"{col} (feature)")
        i += 1
        if i % 4 == 0:
            i += 1

    # Creating output subplot
    plt.subplot(1, col_count, col_count)
    sns.countplot(x = data[target_col])
    plt.title(f"{target_col} (target)")

    # Create proper legends
    zero_patch = mpatches.Patch(color = "white", label="0: No")
    one_patch = mpatches.Patch(color = "white", label="1: Yes")
    plt.figlegend(handles=[zero_patch, one_patch], loc="upper right")

    # setup the layout to fit in the figure
    plt.tight_layout(pad=1, h_pad=0.5, w_pad=0.5) 
    # Save the file with proper dpi
    plt.savefig(fname=f"{output_folder}/input_description{('_' if suffix else '') + suffix}.png", format="png", dpi = fig.dpi)
    logger.log(f"##################### {output_folder}/input_description{('_' if suffix else '') + suffix}.png file saved ###################\n")


def extract_train_test(data : pd.DataFrame, logger : Logger = None) -> list:
    # Seperate features and labels
    X = data.drop(columns=target_col)
    y = data.loc[:, target_col]

    # split train and test datasets
    train_test_sets = train_test_split(X, y, test_size=0.25, random_state=0)

    # Evaluating the proportion of target variable in train and test sets
    X_train, X_test, y_train, y_test = train_test_sets

    y_value_counts = data.value_counts(target_col)
    y_train_value_counts = pd.DataFrame(y_train).value_counts(target_col)
    y_test_value_counts = pd.DataFrame(y_test).value_counts(target_col)

    # Comparing proportion of target variable classes of train and test sets to the whole dataset
    logger.log(f"\nThe proportion of 'yes' in the whole dataset: {y_value_counts[1]/(y_value_counts[0]+y_value_counts[1])}")
    logger.log(f"The proportion of 'yes' in the y_train set: {y_train_value_counts[1]/(y_train_value_counts[0]+y_train_value_counts[1])}")
    logger.log(f"The proportion of 'yes' in the y_test set: {y_test_value_counts[1]/(y_test_value_counts[0]+y_test_value_counts[1])}")

    return train_test_sets

def hyperparameter_tuning_classification(data : pd.DataFrame, train_test_sets : list, estimator, param : dict, logger : Logger = None) -> dict: 
    logger.log("############ Hyperparameter tuning to optimized model parameters ############\n")
    # grid search to find best classifier parameters (Hyperparameter tuning)
    
    # Extract train and test sets
    X_train, X_test, y_train, y_test = train_test_sets
    # Some parameters do not match with each other. This code avoids unnecessary warnings
    warnings.filterwarnings("ignore")

    # Create and train gridsearch
    grid = GridSearchCV(estimator, param_grid=param, refit=True, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)

    # Find the best paramters
    grid_best_params = grid.best_params_
    grid_best_score = grid.best_score_

    logger.log(f"Best Parameters: {grid_best_params}")
    logger.log(f"Best Score: {grid_best_score}")

    return grid_best_params


def classification_decisiontree(data : pd.DataFrame, train_test_sets : list, grid_best_params : dict = {}, suffix : str = "", output_folder : str = "output", logger : Logger = None):
    logger.log("################ classification by decision tree #################\n")
    # classification by decision tree
    
    # grid_best_params is not empty when we are classifying after the optimization
    if grid_best_params:
        clf = DecisionTreeClassifier(max_depth=grid_best_params["max_depth"], min_samples_leaf=grid_best_params["min_samples_leaf"], min_samples_split=grid_best_params["min_samples_split"], random_state = grid_best_params["random_state"], max_features = grid_best_params["max_features"])
    else:
        clf = DecisionTreeClassifier()

    # extract train and test sets from the train_test_sets list
    X_train, X_test, y_train, y_test = train_test_sets

    # train the model
    clf.fit(X_train, y_train)

    # Predict test dataset via model
    y_pred = clf.predict(X_test)

    # # Exporting decesion tree to png file
    # export_tree(clf, suffix, clf.feature_names_in_, output_folder, logger)

    # Evaluate the model
    evaluate_model_decisiontree(clf=clf, data=data, y_test=y_test, y_pred=y_pred, suffix="After Optimization" if grid_best_params else "Before Optimization", output_folder=output_folder, logger=logger)


def evaluate_model_decisiontree(clf : DecisionTreeClassifier, data : pd.DataFrame, y_test : pd.DataFrame, y_pred : pd.DataFrame, suffix : str = "", output_folder : str = "output", logger : Logger = None):
    # Evaluating decision tree model
    logger.log(f"################ Evaluating the decision tree model{'(' + suffix + ')' if suffix else ''} #################")
    # set the figure resolution and dpi
    fig = plt.figure(figsize=(16, 9), dpi=600)

    # Extract confusion matrix and plot its heatmap
    conf_matrix = confusion_matrix(y_test, y_pred)
    logger.log(f"Confusion matrix:\n{conf_matrix}\n")
    sns.heatmap(conf_matrix.reshape(-1,2),annot=True, fmt=',d')
    # Save the file with proper dpi
    plt.savefig(fname=f"{output_folder}/confusion_matrix{('_' if suffix else '') + suffix}.png", format="png", dpi = fig.dpi)
    logger.log(f"##################### {output_folder}/confusion_matrix{('_' if suffix else '') + suffix}.png file saved ###################\n")

    # log the classification report including class-specific precision, recall, f1-score and their macro avg and weighted avg plus model accuracy
    logger.log(f"Classification Report:\n{classification_report(y_test, y_pred,zero_division=0)}")

    # Extract model feature_importances and plot their barh
    # Creating a dataframe based on the feature importances
    feature_importances = pd.DataFrame(clf.feature_importances_,index=clf.feature_names_in_, columns=["feature_importances"])
    feature_importances = feature_importances.sort_values("feature_importances",axis="index")
    logger.log(f"Importance of features:\n{feature_importances}")
    plt.clf()
    # Create a bar plot to compare feature importances visually
    plt.barh(feature_importances.index, feature_importances.iloc[:,0])
    plt.xlabel("feature_importances")
    # Save the file with proper dpi
    plt.savefig(fname=f"{output_folder}/feature_importances{('_' if suffix else '') + suffix}.png", format="png", dpi = fig.dpi)
    logger.log(f"##################### {output_folder}/feature_importances{('_' if suffix else '') + suffix}.png file saved ###################\n")

    logger.log(f"Classes are: {clf.classes_}")


def export_tree(clf : DecisionTreeClassifier, feature_names_in_ : np.ndarray, suffix : str = "", output_folder : str = "output", logger : Logger = None):
    # export decision tree into dot file notaion
    dot_data = export_graphviz(clf, out_file =None ,feature_names=feature_names_in_,class_names=["No","Yes"], impurity=True, filled=True, rounded=True)
    # create png file based on dot data
    graph = graphviz.Source(dot_data, format="png") 
    # render and save png file
    graph.render(filename=f"{output_folder}/decision_tree{('_' if suffix else '') + suffix}", view=False)
    logger.log(f"##################### {output_folder}/decision_tree{('_' if suffix else '') + suffix}.png file saved #################################\n")


def classification_randomforest(data : pd.DataFrame, train_test_sets : list, grid_best_params : dict = {}, suffix : str = "", output_folder : str = "output", logger : Logger = None):
    logger.log("################ classification by random forest #################\n")
    # classification by random forest

    # grid_best_params is not empty when we are classifying after the optimization
    if grid_best_params:
        clf = RandomForestClassifier(max_depth=grid_best_params["max_depth"], min_samples_leaf=grid_best_params["min_samples_leaf"], min_samples_split=grid_best_params["min_samples_split"], random_state = grid_best_params["random_state"], max_features = grid_best_params["max_features"])
    else:
        clf = RandomForestClassifier()

    # extract train and test sets from the train_test_sets list
    X_train, X_test, y_train, y_test = train_test_sets

    # train the model
    clf.fit(X_train, y_train)

    # Predict test dataset via model
    y_pred = clf.predict(X_test)

    # # Exporting decesion trees of the random forest to png file
    # est_count = 1
    # for estimator in clf.estimators_:
    #     export_tree(estimator, clf.feature_names_in_, suffix + "_" + str(est_count), output_folder, logger)
    #     est_count += 1

    # Evaluate the model
    evaluate_model_randomforest(clf=clf, data=data, y_test=y_test, y_pred=y_pred, suffix="After Optimization" if grid_best_params else "Before Optimization", output_folder=output_folder, logger=logger)


def evaluate_model_randomforest(clf : DecisionTreeClassifier, data : pd.DataFrame, y_test : pd.DataFrame, y_pred : pd.DataFrame, suffix : str = "", output_folder : str = "output", logger : Logger = None):
    # Evaluating random forest model
    logger.log(f"################ Evaluating the random forest model{'(' + suffix + ')' if suffix else ''} #################")
    # set the figure resolution and dpi
    fig = plt.figure(figsize=(16, 9), dpi=600)

    # Extract confusion matrix and plot its heatmap
    conf_matrix = confusion_matrix(y_test, y_pred)
    logger.log(f"Confusion matrix:\n{conf_matrix}\n")
    sns.heatmap(conf_matrix.reshape(-1,2),annot=True, fmt=',d')
    # Save the file with proper dpi
    plt.savefig(fname=f"{output_folder}/confusion_matrix{('_' if suffix else '') + suffix}.png", format="png", dpi = fig.dpi)
    logger.log(f"##################### {output_folder}/confusion_matrix{('_' if suffix else '') + suffix}.png file saved ###################\n")

    # log the classification report including class-specific precision, recall, f1-score and their macro avg and weighted avg plus model accuracy
    logger.log(f"Classification Report:\n{classification_report(y_test, y_pred,zero_division=0)}")

    # Extract model feature_importances and plot their barh
    # Creating a dataframe based on the feature importances
    feature_importances = pd.DataFrame(clf.feature_importances_,index=clf.feature_names_in_, columns=["feature_importances"])
    feature_importances = feature_importances.sort_values("feature_importances",axis="index")
    logger.log(f"Importance of features:\n{feature_importances}")
    plt.clf()
    # Create a bar plot to compare feature importances visually
    plt.barh(feature_importances.index, feature_importances.iloc[:,0])
    plt.xlabel("feature_importances")
    # Save the file with proper dpi
    plt.savefig(fname=f"{output_folder}/feature_importances{('_' if suffix else '') + suffix}.png", format="png", dpi = fig.dpi)
    logger.log(f"##################### {output_folder}/feature_importances{('_' if suffix else '') + suffix}.png file saved ###################\n")

    logger.log(f"Classes are: {clf.classes_}")


def classification_logisticregression(data : pd.DataFrame, train_test_sets : list, grid_best_params : dict = {}, suffix : str = "", output_folder : str = "output", logger : Logger = None):
    logger.log("################ classification by logistic regression #################\n")
    # classification by logistic regression

    # grid_best_params is not empty when we are classifying after the optimization
    if grid_best_params:
        clf = LogisticRegression(solver=grid_best_params["solver"], penalty=grid_best_params["penalty"], C=grid_best_params["C"], max_iter=grid_best_params["max_iter"], random_state=grid_best_params["random_state"])
    else:
        clf = LogisticRegression()

    # extract train and test sets from the train_test_sets list
    X_train, X_test, y_train, y_test = train_test_sets

    # train the model
    clf.fit(X_train, y_train)

    # Predict test dataset via model
    y_pred = clf.predict(X_test)

    # Evaluate the model
    evaluate_model_logisticregression(clf=clf, data=data, y_test=y_test, y_pred=y_pred, suffix="After Optimization" if grid_best_params else "Before Optimization", output_folder=output_folder, logger=logger)


def evaluate_model_logisticregression(clf : LogisticRegression, data : pd.DataFrame, y_test : pd.DataFrame, y_pred : pd.DataFrame, suffix : str = "", output_folder : str = "output", logger : Logger = None):
    # Evaluating logistic regression model
    logger.log(f"################ Evaluating the logistic regression model{'(' + suffix + ')' if suffix else ''} #################")
    # set the figure resolution and dpi
    fig = plt.figure(figsize=(16, 9), dpi=600)

    # Extract confusion matrix and plot its heatmap
    conf_matrix = confusion_matrix(y_test, y_pred)
    logger.log(f"Confusion matrix:\n{conf_matrix}\n")
    sns.heatmap(conf_matrix.reshape(-1,2),annot=True, fmt=',d')
    # Save the file with proper dpi
    plt.savefig(fname=f"{output_folder}/confusion_matrix{('_' if suffix else '') + suffix}.png", format="png", dpi = fig.dpi)
    logger.log(f"##################### {output_folder}/confusion_matrix{('_' if suffix else '') + suffix}.png file saved ###################\n")

    # Extract model coefficients and plot their barh
    # Creating a dataframe based on the coefficients
    coef = pd.DataFrame(clf.coef_[0],index=clf.feature_names_in_, columns=["Coefficient"])
    coef = coef.sort_values("Coefficient",axis="index")
    logger.log(f"Coefficient of features:\n{coef}")
    plt.clf()
    # Create a bar plot to compare coefficients visually
    plt.barh(coef.index, coef.iloc[:,0])
    plt.xlabel("Coefficient")
    # Save the file with proper dpi
    plt.savefig(fname=f"{output_folder}/Coefficients{("_" if suffix else '') + suffix}.png", format="png", dpi = fig.dpi)
    logger.log(f"##################### {output_folder}/Coefficient{("_" if suffix else '') + suffix}.png file saved ###################\n")

    # log the classification report including class-specific precision, recall, f1-score and their macro avg and weighted avg plus model accuracy
    logger.log(f"Classification Report:\n{classification_report(y_test, y_pred,zero_division=0)}")



def classification_knn(data : pd.DataFrame, train_test_sets : list, grid_best_params : dict = {}, suffix : str = "", output_folder : str = "output", logger : Logger = None):
    logger.log("################ classification by k-nearest neighbour #################\n")
    # classification by k-nearest neighbour

    # grid_best_params is not empty when we are classifying after the optimization
    if grid_best_params:
        clf = KNeighborsClassifier(n_neighbors=grid_best_params["n_neighbors"], weights=grid_best_params["weights"], metric=grid_best_params["metric"])
    else:
        clf = KNeighborsClassifier()

    # Extract train and test sets from the train_test_sets list
    X_train, X_test, y_train, y_test = train_test_sets

    # train the model
    clf.fit(X_train, y_train)

    # Predict test dataset via model
    y_pred = clf.predict(X_test)

    # Evaluate the model
    evaluate_model_knn(clf=clf, data=data, y_test=y_test, y_pred=y_pred, suffix="After Optimization" if grid_best_params else "Before Optimization", output_folder=output_folder, logger=logger)


def evaluate_model_knn(clf : LogisticRegression, data : pd.DataFrame, y_test : pd.DataFrame, y_pred : pd.DataFrame, suffix : str = "", output_folder : str = "output", logger : Logger = None):
    # Evaluating k-nearest neighbour model
    logger.log(f"################ Evaluating the knn model{'(' + suffix + ')' if suffix else ''} #################")
    # set the figure resolution and dpi
    fig = plt.figure(figsize=(16, 9), dpi=600)

    # Extract confusion matrix and plot its heatmap
    conf_matrix = confusion_matrix(y_test, y_pred)
    logger.log(f"Confusion matrix:\n{conf_matrix}\n")
    sns.heatmap(conf_matrix.reshape(-1,2),annot=True, fmt=',d')
    # Save the file with proper dpi
    plt.savefig(fname=f"{output_folder}/confusion_matrix{('_' if suffix else '') + suffix}.png", format="png", dpi = fig.dpi)
    logger.log(f"##################### {output_folder}/confusion_matrix{('_' if suffix else '') + suffix}.png file saved ###################\n")
    
    # log the classification report including class-specific precision, recall, f1-score and their macro avg and weighted avg plus model accuracy
    logger.log(f"Classification Report:\n{classification_report(y_test, y_pred,zero_division=0)}")


def main() -> int:
    # Folder path to save outputs
    output_folder = "output"
    logger = Logger()

    # Load dataset 
    # SAMAN TEYMOURI FDA WRITTEN EXERCISE
    df = load_data("input/dataset.csv", logger=logger)

    # describe dataset characteristics
    describe_data(df, logger=logger)

    # Show data exploration
    plot_pair(df, "before_data_cleaning", output_folder=output_folder,logger=logger)

    # Plot before data cleaning
    plot_hist(df, "before_data_cleaning", output_folder, logger=logger)

    # Print elapsed time for the below job
    logger.print_elapsed_time("loading and describing the data")

    logger.log("#################### Data cleaning and dataset changes ###################")
    # Data Cleaning Steps

    # Handle missing values
    df = handle_missing_values(df, logger=logger)

    # Find duplicates and remove them
    df = handle_duplicate_values(df, logger=logger)

    # Find outliers and remove them
    df = handle_outlier_values(df, output_folder, logger=logger)

    # Encode data
    df = encode_data(df, logger=logger)

    # describe dataset characteristics
    describe_data(df, logger=logger, suffix="cleaned (except scaling step)")

    # Show correlation between features with each other and with the target variable
    plot_correlation(df, output_folder=output_folder,logger=logger)

    # Show data exploration after data cleaning
    plot_pair(df, "after_data_cleaning", output_folder=output_folder,logger=logger)

    # Plot histograms after data cleaning
    plot_hist(df, "after_data_cleaning", output_folder, logger=logger)
    
    # # Scale data
    df = scale_data(df, logger=logger)

    # Print elapsed time for the below job
    logger.print_elapsed_time("cleaning the data")

    # Extracting train and test sets
    train_test_sets = extract_train_test(df, logger=logger)

    # ================== Modeling via decision tree ================
    # Perform classification using decision tree (before optimization)
    logger.log("#################### Modeling via decision tree ###################")

    classification_decisiontree(data=df, train_test_sets=train_test_sets, output_folder=output_folder + "/decisiontree_model/before_optimization",logger=logger)

    # Print elapsed time for the below job
    logger.print_elapsed_time("classification via decision tree before optimization")

    # Optimizing the parameters of the decision tree model using hyperparameter tuning
    grid_best_params = hyperparameter_tuning_classification(data=df, train_test_sets=train_test_sets, estimator=DecisionTreeClassifier(), param = {
        "max_depth": [10,50,100,200,None],
        "min_samples_leaf": [1, 2],
        "min_samples_split": [2, 3, 4],
        "random_state": [0, 1, 10, 20, 42, None],
        "max_features" : ["sqrt", "log2", None]
    }, logger=logger)

    # Print elapsed time for the below job
    logger.print_elapsed_time("hyperparameter tuning of decision tree")

    # Perform classification using decision tree (after optimization)
    classification_decisiontree(data=df, train_test_sets=train_test_sets, grid_best_params=grid_best_params, output_folder=output_folder + "/decisiontree_model/after_optimization",logger=logger)

    # Print elapsed time for the below job
    logger.print_elapsed_time("classification via decision tree after optimization")
    
    # ================== Modeling via random forest ================
    # Perform classification using random forest (before optimization)
    logger.log("#################### Modeling via random forest ###################")

    classification_randomforest(data=df, train_test_sets=train_test_sets, output_folder=output_folder + "/randomforest_model/before_optimization",logger=logger)

    # Print elapsed time for the below job
    logger.print_elapsed_time("classification via random forest before optimization")

    # Optimizing the parameters of the random forest model using hyperparameter tuning
    grid_best_params = hyperparameter_tuning_classification(data=df, train_test_sets=train_test_sets, estimator=RandomForestClassifier(), param = {
        "max_depth": [10,50,100,200,None],
        "min_samples_leaf": [1, 2],
        "min_samples_split": [2, 3, 4],
        "random_state": [0, 1, 10, 20, 42, None],
        "max_features" : ["sqrt", "log2"]
    }, logger=logger)

    # Print elapsed time for the below job
    logger.print_elapsed_time("hyperparameter tuning of random forest")

    # Perform classification using random forest (after optimization)
    classification_randomforest(data=df, train_test_sets=train_test_sets, grid_best_params=grid_best_params, output_folder=output_folder + "/randomforest_model/after_optimization",logger=logger)

    # Print elapsed time for the below job
    logger.print_elapsed_time("classification via random forest after optimization")

    # ================== Modeling via logistic regression ================
    # Perform classification using logistic regression (before optimization)
    logger.log("#################### Modeling via logistic regression ###################")

    classification_logisticregression(data=df, train_test_sets=train_test_sets, output_folder=output_folder + "/logisticregression_model/before_optimization",logger=logger)

    # Print elapsed time for the below job
    logger.print_elapsed_time("classification via logistic regression before optimization")

    # Optimizing the parameters of the logistic regression model using hyperparameter tuning
    grid_best_params = hyperparameter_tuning_classification(data=df, train_test_sets=train_test_sets, estimator=LogisticRegression(), param = {
        "solver": ["newton-cg", "newton-cholesky", "lbfgs", "liblinear", "sag", "saga"],
        "penalty": ["l1", "l2", "elasticnet", "None"],
        "C": [100, 10, 1, 0.1, 0.01],
        "max_iter"  : [100,1000,2500,5000],
        "random_state": [0, 1, 10, 20, 42, None],
    }, logger=logger)

    # Print elapsed time for the below job
    logger.print_elapsed_time("hyperparameter tuning of logistic regression")

    # Perform classification using logistic regression (after optimization)
    classification_logisticregression(data=df, train_test_sets=train_test_sets, grid_best_params=grid_best_params, output_folder=output_folder + "/logisticregression_model/after_optimization",logger=logger)

    # Print elapsed time for the below job
    logger.print_elapsed_time("classification via logistic regression after optimization")

    # ================== Modeling via k-nearest neighbor ================
    # Perform classification using k-nearest neighbour (before optimization)
    logger.log("#################### Modeling via k-nearest neighbour ###################")

    classification_knn(data=df, train_test_sets=train_test_sets, output_folder=output_folder + "/knn_model/before_optimization",logger=logger)

    # Print elapsed time for the below job
    logger.print_elapsed_time("classification via K-nearest neighbour before optimization")

    # Optimizing the parameters of the k-nearest neighbour model using hyperparameter tuning
    grid_best_params = hyperparameter_tuning_classification(data=df, train_test_sets=train_test_sets, estimator=KNeighborsClassifier(), param = {
        "n_neighbors": [3,5,7,9,11,13,15],
        "weights": ["uniform","distance"],
        "metric": ["minkowski","euclidean","manhattan"]
    }, logger=logger)
    
    # Print elapsed time for the below job
    logger.print_elapsed_time("hyperparameter tuning of K-nearest neighbour")

    # Perform classification using k-nearest neighbour (after optimization)
    classification_knn(data=df, train_test_sets=train_test_sets, grid_best_params=grid_best_params, output_folder=output_folder + "/knn_model/after_optimization",logger=logger)

    # Print elapsed time for the below job
    logger.print_elapsed_time("classification via K-nearest neighbour after optimization")

    # Save log file
    logger.save_file(f"{output_folder}/output_log.txt")

    print("\n##########################################################################\n" +
            f"Please check <{output_folder}> folder in the program path for outputs.\n" +
            "##########################################################################\n")

    return 0

if __name__ == "__main__":
    main()
