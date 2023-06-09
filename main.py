import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import preprocessing
import statsmodels.api as sm


def dataPre():
    # Load Data
    file_path = "dataset.csv"
    df_draft = pd.read_csv(file_path)

    # Change time
    df_draft['time'] = pd.to_datetime(df_draft['time'])
    df_draft['time'] = df_draft['time'].values.astype(np.int64) // 10 ** 9

    # Checking missing values
    print()
    print("------------Checking missing values-------------")
    print(df_draft.isnull().sum())
    # Removing missing values
    df = df_draft.dropna()
    print()
    print("------------After removing missing values-------------")
    print(df.isnull().sum())

    # Drop Ammonia nitrogen
    df = df.drop(columns='Ammonia nitrogen')

    # For checking correlation between variables
    correlationHeatmap(df)
    # Dropping columns based on correlation
    df = df.drop(columns=['Dissolved Oxygen(% Sat)', 'Total dissolved solids', 'salinity'])

    # Checking VIF of remaining features
    vif(df)
    # Dropping columns based on vif
    df = df.drop(columns='time')

    # Data Scaling
    df_final=dataScale(df)
    return df_final

def dataAnalysis_MLR():
    df_final = dataPre()
    # Data Split Training-Test
    x = df_final[['longitude', 'latitude', 'chlorophyll', 'electrical conductivity', 'Low Frequency Water Depth(m)',
                  'Dissolved oxygen(mg/L)',
                  'turbidity', 'temperature', 'PH value', 'PH value(mv)']]
    y = df_final[['Phycotin']]
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3)

    md = model_MLR(x_train, y_train)
    validate_MLR(md, x_train, x_test, y_train, y_test)
    visualization(df_final)

def dataAnalysis_DT():
    df_final = dataPre()
    # Data Split Training-Test
    x = df_final[['longitude', 'latitude', 'chlorophyll', 'electrical conductivity', 'Low Frequency Water Depth(m)',
                  'Dissolved oxygen(mg/L)',
                  'turbidity', 'temperature', 'PH value', 'PH value(mv)']]
    y = df_final[['Phycotin']]
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3)

    # model
    md = model_DT(x_train, y_train)
    
    # validate
    validate_DT(md, x_train, x_test, y_train, y_test)

    # feature Importance
    featureImpo(md)


def correlationHeatmap(df):
    # Correlation
    corr = df.corr(method='pearson')
    df_heatmap = sb.heatmap(corr, cbar=True, annot=True, annot_kws={'size': 10}, fmt='.2f', square=True, cmap='Blues')
    plt.show()

def vif(df):
    X_train = df[
        ['longitude', 'latitude', 'time', 'chlorophyll', 'electrical conductivity', 'Low Frequency Water Depth(m)',
         'Dissolved oxygen(mg/L)',
         'turbidity', 'temperature', 'PH value', 'PH value(mv)']]
    vif = pd.DataFrame()
    vif['VIF_Factor'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
    vif['Feature'] = X_train.columns
    print()
    print("------------VIF--------------")
    print(vif)

def dataScale(df):
    # Data Scaling
    columns = ['longitude', 'latitude', 'chlorophyll', 'electrical conductivity', 'Low Frequency Water Depth(m)',
         'Dissolved oxygen(mg/L)', 'turbidity', 'temperature', 'PH value', 'PH value(mv)']
    scaler = StandardScaler()
    scaler.fit(df[columns])
    df[columns] = scaler.transform(df[columns])
    df_final = pd.DataFrame(df)
    df_final.columns = df.columns
    return df_final

def model_MLR(x_train, y_train):
    # Create a Model
    mlr = LinearRegression()
    # slope and interception
    model = mlr.fit(x_train, y_train)
    # a1....a10
    print()
    print('------------coefficient--------------')
    print(model.coef_)
    # a0
    print()
    print('------------intercept--------------')
    print(model.intercept_)
    return model

def model_DT(x_train, y_train):

    Tree = DecisionTreeRegressor(max_depth=5)
    model = Tree.fit(x_train, y_train)
    
    #show tree
    plt.figure(figsize=(5, 4), dpi=200)
    plot_tree(model, filled=True)
    plt.show()
    
    return model


def validate_MLR(model, x_train, x_test, y_train, y_test):
    y_predict = model.predict(x_test)
    y_predict = pd.DataFrame(y_predict)

    # Generating table of Actual Value vs Predicted Value
    y_compare = [y_test, y_predict]
    compare_result = pd.concat(y_compare, axis=1, join='inner')
    compare_result.rename(columns={'Phycotin': 'Actual Value', 0: 'Predicted Value'}, inplace=True)
    print()
    print('--------------Compare Actual Value with Predicted Value----------------')
    print(compare_result)

    # Drawing scatter plot of Actual Value vs Predicted Value
    plt.scatter(y_test, y_predict, alpha=0.4)
    plt.xlabel("Actual Phycotin")
    plt.ylabel("Predicted Phycotin")
    plt.title("MULTIPLE LINEAR REGRESSION")
    plt.show()

    # Calculating accuracy
    accuracy = model.score(x_train, y_train)
    print()
    print('--------------Accuracy----------------')
    print(accuracy)
    print(accuracy*100,'%')

def validate_DT(model, x_train, x_test, y_train, y_test):
    y_predict = model.predict(x_test)
    y_predict = pd.DataFrame(y_predict)

    # Generating table of Actual Value vs Predicted Value
    y_compare = [y_test, y_predict]
    compare_result = pd.concat(y_compare, axis=1, join='inner')
    compare_result.rename(columns={'Phycotin': 'Actual Value', 0: 'Predicted Value'}, inplace=True)
    print()
    print('--------------Compare Actual Value with Predicted Value----------------')
    print(compare_result)

    # Calculating accuracy
    print()
    print('--------------Accuracy----------------')
    y_predict = model.predict(x_test)
    mse = mean_squared_error(y_test, y_predict)
    rmse = np.sqrt(mse)
    r2_points = r2_score(y_test, y_predict)
    print("RMSE:", rmse)
    print("R^2:", r2_points)

def featureImpo(model):
    columns = ['longitude', 'latitude', 'chlorophyll', 'electrical conductivity', 'Low Frequency Water Depth(m)',
               'Dissolved oxygen(mg/L)', 'turbidity', 'temperature', 'PH value', 'PH value(mv)']
    feature_importance_values = model.feature_importances_
    # Sorting
    feature_importances = pd.Series(feature_importance_values, index=columns)
    feature_top3 = feature_importances.sort_values(ascending=False)[:5]

    plt.figure(figsize=[8, 6])
    plt.title('Feature Importances Top 5')
    sb.barplot(x=feature_top3, y=feature_top3.index)
    plt.show()

def visualization(df_final):
    # 'latitude' & 'Phycotin'
    plt.scatter(df_final[['latitude']], df_final[['Phycotin']], alpha=0.4)
    plt.xlabel('latitude')
    plt.ylabel("Phycotin")
    plt.title("Relationship Between Latitude and Phycotin")
    plt.show()

    # 'electrical conductivity' & 'Phycotin'
    plt.scatter(df_final[['electrical conductivity']], df_final[['Phycotin']], alpha=0.4)
    plt.xlabel('electrical conductivity')
    plt.ylabel("Phycotin")
    plt.title("Relationship Between Electrical Conductivity and Phycotin")
    plt.show()

    # 'Low Frequency Water Depth(m)' & 'Phycotin'
    plt.scatter(df_final[['Low Frequency Water Depth(m)']], df_final[['Phycotin']], alpha=0.4)
    plt.xlabel('Low Frequency Water Depth(m)')
    plt.ylabel("Phycotin")
    plt.title("Relationship Between Low Frequency Water Depth and Phycotin")
    plt.show()

    # 'Dissolved oxygen(mg/L)' & 'Phycotin'
    plt.scatter(df_final[['Dissolved oxygen(mg/L)']], df_final[['Phycotin']], alpha=0.4)
    plt.xlabel('Dissolved oxygen(mg/L)')
    plt.ylabel("Phycotin")
    plt.title("Relationship Between Dissolved Oxygen and Phycotin")
    plt.show()

    # 'temperature' & 'Phycotin'
    plt.scatter(df_final[['temperature']], df_final[['Phycotin']], alpha=0.4)
    plt.xlabel('temperature')
    plt.ylabel("Phycotin")
    plt.title("Relationship Between Temperature and Phycotin")
    plt.show()

    # 'PH value' & 'Phycotin'
    plt.scatter(df_final[['PH value']], df_final[['Phycotin']], alpha=0.4)
    plt.xlabel('PH value')
    plt.ylabel("Phycotin")
    plt.title("Relationship Between PH Value and Phycotin")
    plt.show()

    # 'PH value(mv)' & 'Phycotin'
    plt.scatter(df_final[['PH value(mv)']], df_final[['Phycotin']], alpha=0.4)
    plt.xlabel("PH value(mv)")
    plt.ylabel("Phycotin")
    plt.title("Relationship Between PH Value(mv) and Phycotin")
    plt.show()

    # 'chlorophyll' & 'Phycotin'
    plt.scatter(df_final[['chlorophyll']], df_final[['Phycotin']], alpha=0.4)
    plt.xlabel("chlorophyll")
    plt.ylabel("Phycotin")
    plt.title("Relationship Between Chlorophyll and Phycotin")
    plt.show()

    # 'turbidity' & 'Phycotin'
    plt.scatter(df_final[['turbidity']], df_final[['Phycotin']], alpha=0.4)
    plt.xlabel("turbidity")
    plt.ylabel("Phycotin")
    plt.title("Relationship Between Turbidity and Phycotin")
    plt.show()

    # 'longitude' & 'Phycotin'
    plt.scatter(df_final[['longitude']], df_final[['Phycotin']], alpha=0.4)
    plt.xlabel('longitude')
    plt.ylabel("Phycotin")
    plt.title("Relationship Between Longitude and Phycotin")
    plt.show()


if __name__ == "__main__":
    model = input("Please choose a model\n"
                  "Type 'MLR' for Multiple Linear Regression\n"
                  "Type 'DT' for Decision Trees\n"
                  "-> ")
    if (model == 'MLR'):
        dataAnalysis_MLR()
    elif (model == 'DT'):
        dataAnalysis_DT()

