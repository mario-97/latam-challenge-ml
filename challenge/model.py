import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Tuple, Union, List
from datetime import datetime
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb
from xgboost import plot_importance
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')
from typing import Tuple, Union, List

class DelayModel:

    def __init__(self, model):
        self._model = model  # Model should be saved in this attribute.

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare raw data for prediction.

        Args:
            data (pd.DataFrame): raw data.

        Returns:
            pd.DataFrame: preprocessed data.
        """
        # Your preprocessing logic here
        features, target = preprocess_encode(data)
        return features, target

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.Series): target.

        Returns:
            None
        """
        # Your fitting logic here 
        self._model.fit(features, target)

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            List[int]: predicted targets.
        """
        # Your prediction logic here
        self._model.predict(features) 

    
def get_period_day(date):
    date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
    morning_min = datetime.strptime("05:00", '%H:%M').time()
    morning_max = datetime.strptime("11:59", '%H:%M').time()
    afternoon_min = datetime.strptime("12:00", '%H:%M').time()
    afternoon_max = datetime.strptime("18:59", '%H:%M').time()
    evening_min = datetime.strptime("19:00", '%H:%M').time()
    evening_max = datetime.strptime("23:59", '%H:%M').time()
    night_min = datetime.strptime("00:00", '%H:%M').time()
    night_max = datetime.strptime("4:59", '%H:%M').time()
    
    if(date_time > morning_min and date_time < morning_max):
        return 'mañana'
    elif(date_time > afternoon_min and date_time < afternoon_max):
        return 'tarde'
    elif(
        (date_time > evening_min and date_time < evening_max) or
        (date_time > night_min and date_time < night_max)
    ):
        return 'noche'
    
def is_high_season(fecha):
    fecha_año = int(fecha.split('-')[0])
    fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
    range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year = fecha_año)
    range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year = fecha_año)
    range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year = fecha_año)
    range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year = fecha_año)
    range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year = fecha_año)
    range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year = fecha_año)
    range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year = fecha_año)
    range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year = fecha_año)
    
    if ((fecha >= range1_min and fecha <= range1_max) or 
        (fecha >= range2_min and fecha <= range2_max) or 
        (fecha >= range3_min and fecha <= range3_max) or
        (fecha >= range4_min and fecha <= range4_max)):
        return 1
    else:
        return 0
    
def get_min_diff(data):
    fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
    fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
    min_diff = ((fecha_o - fecha_i).total_seconds())/60
    return min_diff

def get_rate_from_column(data, column):
    delays = {}
    for _, row in data.iterrows():
        if row['delay'] == 1:
            if row[column] not in delays:
                delays[row[column]] = 1
            else:
                delays[row[column]] += 1
    total = data[column].value_counts().to_dict()
    
    rates = {}
    for name, total in total.items():
        if name in delays:
            rates[name] = round(total / delays[name], 2)
        else:
            rates[name] = 0
            
    return pd.DataFrame.from_dict(data = rates, orient = 'index', columns = ['Tasa (%)'])
    
def data_analysis(data):
    flights_by_airline = data['OPERA'].value_counts()
    plt.figure(figsize = (10, 2))
    sns.set(style="darkgrid")
    sns.barplot(x=flights_by_airline.index, y=flights_by_airline.values, alpha=0.9)
    plt.title('Flights by Airline')
    plt.ylabel('Flights', fontsize=12)
    plt.xlabel('Airline', fontsize=12)
    plt.xticks(rotation=90)
    plt.show()

    flights_by_day = data['DIA'].value_counts()
    plt.figure(figsize = (10, 2))
    sns.set(style = "darkgrid")
    sns.barplot(x=flights_by_day.index, y=flights_by_day.values, color = 'lightblue', alpha=0.8)
    plt.title('Flights by Day')
    plt.ylabel('Flights', fontsize=12)
    plt.xlabel('Day', fontsize=12)
    plt.xticks(rotation=90)
    plt.show()

    flights_by_month = data['MES'].value_counts()
    plt.figure(figsize = (10, 2))
    sns.set(style = "darkgrid")
    sns.barplot(x=flights_by_month.index, y=flights_by_month.values, color = 'lightblue', alpha=0.8)
    plt.title('Flights by Month')
    plt.ylabel('Flights', fontsize=12)
    plt.xlabel('Month', fontsize=12)
    plt.xticks(rotation=90)
    plt.show()

    flights_by_day_in_week = data['DIANOM'].value_counts()
    days = [
        flights_by_day_in_week.index[2], 
        flights_by_day_in_week.index[5], 
        flights_by_day_in_week.index[4], 
        flights_by_day_in_week.index[1], 
        flights_by_day_in_week.index[0], 
        flights_by_day_in_week.index[6], 
        flights_by_day_in_week.index[3]
    ]
    values_by_day = [
        flights_by_day_in_week.values[2], 
        flights_by_day_in_week.values[5], 
        flights_by_day_in_week.values[4], 
        flights_by_day_in_week.values[1], 
        flights_by_day_in_week.values[0], 
        flights_by_day_in_week.values[6], 
        flights_by_day_in_week.values[3]
    ]
    plt.figure(figsize = (10, 2))
    sns.set(style="darkgrid")
    sns.barplot(x=days, y=values_by_day, color = 'lightblue', alpha=0.8)
    plt.title('Flights by Day in Week')
    plt.ylabel('Flights', fontsize=12)
    plt.xlabel('Day in Week', fontsize=12)
    plt.xticks(rotation=90)
    plt.show()

    flights_by_type = data['TIPOVUELO'].value_counts()
    sns.set(style="darkgrid")
    plt.figure(figsize = (10, 2))
    sns.barplot(x=flights_by_type.index, y=flights_by_type.values, alpha=0.9)
    plt.title('Flights by Type')
    plt.ylabel('Flights', fontsize=12)
    plt.xlabel('Type', fontsize=12)
    plt.show()

    flight_by_destination = data['SIGLADES'].value_counts()
    plt.figure(figsize = (10, 2))
    sns.set(style="darkgrid")
    sns.barplot(x=flight_by_destination.index, y=flight_by_destination.values, color = 'lightblue', alpha=0.8)
    plt.title('Flight by Destination')
    plt.ylabel('Flights', fontsize=12)
    plt.xlabel('Destination', fontsize=12)
    plt.xticks(rotation=90)

    plt.show()

def feature_generation(data):
    ### 2.a. Period of Day
    data['period_day'] = data['Fecha-I'].apply(get_period_day)
    ### 2.b. High Season
    data['high_season'] = data['Fecha-I'].apply(is_high_season)
    ### 2.c. Difference in Minutes
    data['min_diff'] = data.apply(get_min_diff, axis = 1)
    ### 2.d. Delay
    threshold_in_minutes = 15
    data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)
     
    return data

def rate_delay(data):
    destination_rate = get_rate_from_column(data, 'SIGLADES')
    destination_rate_values = data['SIGLADES'].value_counts().index
    plt.figure(figsize = (20,5))
    sns.set(style="darkgrid")
    sns.barplot(x=destination_rate_values, y=destination_rate['Tasa (%)'], alpha = 0.75)
    plt.title('Delay Rate by Destination')
    plt.ylabel('Delay Rate [%]', fontsize=12)
    plt.xlabel('Destination', fontsize=12)
    plt.xticks(rotation=90)
    plt.show()

    airlines_rate = get_rate_from_column(data, 'OPERA')
    airlines_rate_values = data['OPERA'].value_counts().index
    plt.figure(figsize = (20,5))
    sns.set(style="darkgrid")
    sns.barplot(x=airlines_rate_values, y=airlines_rate['Tasa (%)'], alpha = 0.75)
    plt.title('Delay Rate by Airline')
    plt.ylabel('Delay Rate [%]', fontsize=12)
    plt.xlabel('Airline', fontsize=12)
    plt.xticks(rotation=90)
    plt.show()

    month_rate = get_rate_from_column(data, 'MES')
    month_rate_value = data['MES'].value_counts().index
    plt.figure(figsize = (20,5))
    sns.set(style="darkgrid")
    sns.barplot(x=month_rate_value, y=month_rate['Tasa (%)'], color = 'blue', alpha = 0.75)
    plt.title('Delay Rate by Month')
    plt.ylabel('Delay Rate [%]', fontsize=12)
    plt.xlabel('Month', fontsize=12)
    plt.xticks(rotation=90)
    plt.ylim(0,10)
    plt.show()

    days_rate = get_rate_from_column(data, 'DIANOM')
    days_rate_value = data['DIANOM'].value_counts().index

    sns.set(style="darkgrid")
    plt.figure(figsize = (20, 5))
    sns.barplot(x=days_rate_value, y=days_rate['Tasa (%)'], color = 'blue', alpha = 0.75)
    plt.title('Delay Rate by Day')
    plt.ylabel('Delay Rate [%]', fontsize=12)
    plt.xlabel('Days', fontsize=12)
    plt.xticks(rotation=90)
    plt.ylim(0,7)
    plt.show()

    high_season_rate = get_rate_from_column(data, 'high_season')
    high_season_rate_values = data['high_season'].value_counts().index

    plt.figure(figsize = (5, 2))
    sns.set(style="darkgrid")
    sns.barplot(x=["no", "yes"], y=high_season_rate['Tasa (%)'])
    plt.title('Delay Rate by Season')
    plt.ylabel('Delay Rate [%]', fontsize=12)
    plt.xlabel('High Season', fontsize=12)
    plt.xticks(rotation=90)
    plt.ylim(0,6)
    plt.show()

    flight_type_rate = get_rate_from_column(data, 'TIPOVUELO')
    flight_type_rate_values = data['TIPOVUELO'].value_counts().index
    plt.figure(figsize = (5, 2))
    sns.set(style="darkgrid")
    sns.barplot(x=flight_type_rate_values, y=flight_type_rate['Tasa (%)'])
    plt.title('Delay Rate by Flight Type')
    plt.ylabel('Delay Rate [%]', fontsize=12)
    plt.xlabel('Flight Type', fontsize=12)
    plt.ylim(0,7)
    plt.show()

    period_day_rate = get_rate_from_column(data, 'period_day')
    period_day_rate_values = data['period_day'].value_counts().index
    plt.figure(figsize = (5, 2))
    sns.set(style="darkgrid")
    sns.barplot(x=period_day_rate_values, y=period_day_rate['Tasa (%)'])
    plt.title('Delay Rate by Period of Day')
    plt.ylabel('Delay Rate [%]', fontsize=12)
    plt.xlabel('Period', fontsize=12)
    plt.ylim(3,7)
    plt.show()  

def preprocess_encode(data):
    # Verificar si 'delay' está en las columnas
    if 'delay' in data.columns:
        # Aleatorizar las filas de las columnas seleccionadas
        shuffled_data = shuffle(data[['OPERA', 'MES', 'TIPOVUELO', 'SIGLADES', 'DIANOM', 'delay']], random_state=111)
        
        # Codificación de columnas con one-hot encoding
        features = pd.concat([
            pd.get_dummies(shuffled_data['OPERA'], prefix='OPERA'),
            pd.get_dummies(shuffled_data['TIPOVUELO'], prefix='TIPOVUELO'), 
            pd.get_dummies(shuffled_data['MES'], prefix='MES')], 
            axis=1
        )
        
        target = shuffled_data['delay']
    else:
        # Aleatorizar las filas de las columnas seleccionadas
        shuffled_data = shuffle(data[['OPERA', 'MES', 'TIPOVUELO', 'SIGLADES', 'DIANOM']], random_state=111)

        # Codificación de columnas con one-hot encoding
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix='OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'), 
            pd.get_dummies(data['MES'], prefix='MES')], 
            axis=1
        )
        
        target = None

    return features, target

def preprocess_importance_balance(xgb_model, y_train):
    ### Feature Importance
    plt.figure(figsize = (10,5))
    plot_importance(xgb_model)
    top_10_features = [
        "OPERA_Latin American Wings", 
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air"
    ]

    ### Data Balance
    n_y0 = len(y_train[y_train == 0])
    n_y1 = len(y_train[y_train == 1])
    scale = n_y0/n_y1
    return top_10_features, scale, n_y0, n_y1

if __name__ == "__main__":
    # Cargar datos
    data = pd.read_csv('../data/data.csv')
    data.info()

    ## 1. Data Analysis: Primer vistazo
    ### ¿Cómo se distribuyen los datos?
    data_analysis(data)

    ## 2. Features Generation
    data = feature_generation(data)

    ## 3. Data Analysis: Segundo vistazo
    ### ¿Cómo es la tasa de retraso entre columnas?
    rate_delay(data)

    ## 4. Training
    ### 4.a. Data Split (Training and Validation)

    # Preprocesamiento
    features, target = preprocess_encode(data)

    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.33, random_state = 42)

    print(f"train shape: {x_train.shape} | test shape: {x_test.shape}")
    print(y_train.value_counts('%')*100)
    print(y_test.value_counts('%')*100)

    ### 4.b. Model Selection
    #### 4.b.i. XGBoost

    print("========================XGBoost========================")
    print("Entrenamiento")
    xgb_model = xgb.XGBClassifier(random_state=1, learning_rate=0.01)
    xgb_model.fit(x_train, y_train)
    print("Prediccion")
    xgboost_y_preds = xgb_model.predict(x_test)
    xgboost_y_preds = [1 if y_pred > 0.5 else 0 for y_pred in xgboost_y_preds]
    print("Reporte")
    confusion_matrix(y_test, xgboost_y_preds)
    print(classification_report(y_test, xgboost_y_preds))

    #### 4.b.ii. Logistic Regression
    print("========================LogisticRegression========================")
    print("Entrenamiento")
    reg_model = LogisticRegression()
    reg_model.fit(x_train, y_train)
    print("Prediccion")
    reg_y_preds = reg_model.predict(x_test)
    print("Reporte")
    confusion_matrix(y_test, reg_y_preds)
    print(classification_report(y_test, reg_y_preds))

    ## 5. Data Analysis: Tercer vistazo
    top_10_features, scale, n_y0, n_y1 = preprocess_importance_balance(xgb_model, y_train)

    ## 6. Training with Improvement
    ### 6.a. Data Split
    x_train2, x_test2, y_train2, y_test2 = train_test_split(features[top_10_features], target, test_size = 0.33, random_state = 42)

    ### 6.b. Model Selection
    #### 6.b.i. XGBoost with Feature Importance and with Balance
    print("========================XGBoost with Feature Importance and with Balance========================")
    print("Entrenamiento")
    xgb_model_2 = xgb.XGBClassifier(random_state=1, learning_rate=0.01, scale_pos_weight = scale)
    xgb_model_2.fit(x_train2, y_train2)
    print("Prediccion")
    xgboost_y_preds_2 = xgb_model_2.predict(x_test2)
    print("Reporte")
    confusion_matrix(y_test2, xgboost_y_preds_2)
    print(classification_report(y_test2, xgboost_y_preds_2))

    #### 6.b.ii. XGBoost with Feature Importance but without Balance
    print("========================XGBoost with Feature Importance and without Balance========================")
    print("Entrenamiento")
    xgb_model_3 = xgb.XGBClassifier(random_state=1, learning_rate=0.01)
    xgb_model_3.fit(x_train2, y_train2)
    print("Prediccion")
    xgboost_y_preds_3 = xgb_model_3.predict(x_test2)
    print("Reporte")
    confusion_matrix(y_test2, xgboost_y_preds_3)
    print(classification_report(y_test2, xgboost_y_preds_3))


    #### 6.b.iii. Logistic Regression with Feature Importante and with Balance
    print("========================LogisticRegression with Feature Importance and with Balance========================")
    print("Entrenamiento")
    reg_model_2 = LogisticRegression(class_weight={1: n_y0/len(y_train), 0: n_y1/len(y_train)})
    reg_model_2.fit(x_train2, y_train2)
    print("Prediccion")
    reg_y_preds_2 = reg_model_2.predict(x_test2)
    print("Reporte")
    confusion_matrix(y_test2, reg_y_preds_2)
    print(classification_report(y_test2, reg_y_preds_2))

    #### 6.b.iv. Logistic Regression with Feature Importante but without Balance
    print("========================LogisticRegression with Feature Importance and without Balance========================")
    print("Entrenamiento")
    reg_model_3 = LogisticRegression()
    reg_model_3.fit(x_train2, y_train2)
    print("Prediccion")
    reg_y_preds_3 = reg_model_3.predict(x_test2)
    print("Reporte")
    confusion_matrix(y_test2, reg_y_preds_3)
    print(classification_report(y_test2, reg_y_preds_3))

    ## 7. Data Science Conclusions
    # AÑADIR MIS CONCLUSIONES EN CHALLENGE.MD
    """ By looking at the results of the 6 trained models, it can be determined:
    - There is no noticeable difference in results between XGBoost and LogisticRegression.
    - Does not decrease the performance of the model by reducing the features to the 10 most important.
    - Improves the model's performance when balancing classes, since it increases the recall of class "1". """

    """ **With this, the model to be productive must be the one that is trained with the top 10 features and class balancing, but which one?** """
