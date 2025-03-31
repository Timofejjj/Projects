import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import int64, float64
from sklearn.preprocessing import StandardScaler, MinMaxScaler


###########################_Вопросы_###########################
#
#   Не понятно почему в данных имеется так много дат вида: 0001-01-01T00:00:00, что это значит?
#   Не понятна строчка: np.Timestamp('0001-01-01T00:00:00')
#
#   В ноутбуке указано построение корреляции для данных с названием df_cleaned,
#   Но как построилась тепловая карта если они не содержат столбца is_real_incedent
#   
################################################################



#######################_Блок 1_#######################
df = pd.read_csv('ml_dataset_20k.csv')
 
#Это первые 20 строк
print(df.head(20))

# Информация о параметрах
print(df.info())

# !      В этих данных целевой параметр - это признак is_real_incedent                         !
# !      1 - означает что событие является реальным, 0 - означает что событие является ложным  !

#######################_Блок 2_#######################
# Целевой парамтер в нших данных, это признак is_real_incedent
# .value_counts() - Возвращает количество уникальных значений в признаке
print(df['is_real_incident'].value_counts())

# Возвращает проценты
print(df['is_real_incident'].value_counts(normalize=True).map(lambda x: f"{x:.2%}"))



#######################_Блок 3_#######################
print("#################Статистические характеристики признаков:#################")
# Метод describe() возвращает статистические характеристики признаков это значит что 
# он возвращает среднее значение, медиану, дисперсию, минимальное и максимальное значение, 25%, 50%, 75% квантили
#
# В выводе будет:
# std - это стандартное отклонение
# mean - среднее значение
# min - минимальное значение
# max - максимальное значение
# 25% - 25% квантиль (25% значений признака меньше этого значения)
# 50% - медиана (50% значений признака меньше этого значения)
# 75% - 75% квантиль (75% значений признака меньше этого значения)
print(df.describe().T)


#######################_Блок 4_#######################
print("#################_Работа с пропущенными значениями:_#################")
# Проверка на пропущенные значения
# isnull() - возвращает True если значение признака равно NaN
# sum() - возвращает количество True

# Посмотрим, есть ли пропущенные значения в каждом признаке
print(f" Исходные пропущенные значения: {df.isnull().sum()}")


# Реализуем проверку на пропущенные значения в каждом признаке и заполним пропущенные значения медианой
for col in df.columns:

    # То есть когда признак имеет пропущенные значения
    if df[col].isnull().sum() > 0:

        # То этой строчкой сразу заполняем пропущенные значения медианой
        df[col] = df[col].fillna(df[col].median())
        print("В столбце {col} заполнили пропуски значением {df[col].median}")    
        
print(df.isnull().sum())    


#######################_Блок 5_#######################

print("#################_Обработка временных признаков:_#################")
# Обработка временных признаков (это признаки, которые имеют дату и время)
# Это признак order_start, с ними будем делать следующее:

# 1) Преобразовать в datetime, то есть столбец order_start будет иметь тип datetime

data_columns = ['last_order_end', 'after_order_start']

print(df['after_order_start'].head(10))
print(df['last_order_end'].head(10))

# Преобразование в datetime
for col in data_columns:
    if col in df.columns:
        # Ошибка erros='coerce' - если в столбце есть неправильный формат даты, то он будет заменен на NaT
        df[col] = pd.to_datetime(df[col], errors='coerce')


# Метод describe() возвращает статистические характеристики признаков это значит что 
# он возвращает среднее значение, медиану, дисперсию, минимальное и максимальное значение, 25%, 50%, 75% квантили
#
# В выводе будет:
# std - это стандартное отклонение
# mean - среднее значение
# min - минимальное значение
print(df['after_order_start'].describe())
print(df['last_order_end'].describe())

# Немого о типе NaT - это специальный тип данных, который означает отсутствие значения
# В строчке 0001-01-01T00:00:00 то есть 0001 год 1 месяц 1 день 0 часов 0 минут 0 секунд

# Почему в столбце last_order_end так много NaT?
# Потому что в столбце last_order_end нет даты, то есть нет ни одного значения

# Обработаем знчения 0001-01-01T00:00:00 в столбце last_order_end спецаальным образом:
# Метод pd.isna() возвращает True если значение признака равно NaT
# Метод shape[0] возвращает количество строк (Количество True)
invalid_dates = pd.isna(df['last_order_end']).shape[0]

# Что делаеет значение ({invalid_dates/len(df):.2%})
#
# Выражение :.2% - это процент от общего количества строк
print(f"\nКоличество строк с датой 0001-01-01: {invalid_dates} ({invalid_dates/len(df):.2%})")

# Напишем обработку дат, которые не соответствуют этому формату:

if invalid_dates > 0:
    # Метод .astype(int) - преобразует признак в число
    # То есть строку df['last_order_end'] != np.Timestamp('0001-01-01T00:00:00') преобразует в число 1 если True и 0 если False
    # Суть этой строчки в том, что мы создаем признак has_previous_order, который равен 1 если в столбце last_order_end есть дата и 0 если нет
    df['has_previous_order'] = (df['last_order_end'] != pd.Timestamp('0001-01-01T00:00:00')).astype(int)


# Зададим дополнительные типы данных для признаков дат:

# dt - это метод, который возвращает объект datetime
# day_of_week - это метод, который возвращает день недели   
#
# 0 — понедельник,
# 1 — вторник,
# 2 — среда,
# 3 — четверг,
# 4 — пятница,
# 5 — суббота,
# 6 — воскресенье.
df['after_order_start_day'] = df['after_order_start'].dt.day_of_week
df['after_order_start_month'] = df['after_order_start'].dt.month
df['is_weekend'] = df['after_order_start'].dt.dayofweek >= 5


#######################_Блок 6_#######################
print("#################_Анализ распределения признаков:_#################")

# Построим функццию, которая строит гистограмму распределения признаков
def plot_distribution(dataframe, n_cols = 3, exclude_cols = None):

    if exclude_cols is None:
        exclude_cols = []

    # Исправляем проверку типов данных
    num_features = [col for col in dataframe.columns 
                    if col not in exclude_cols 
                    and dataframe[col].dtype in [int64, float64]]

    # Тут мы считаем количество строк, которые будут в гистограмме, в каждой строке будет по n_cols гистограмм
    # np.ceil() - округляет вверх
    n_rows = int(np.ceil(len(num_features) / n_cols))


    # Функция subplots() создает массив осей (axes) для графика, визуально: 
    # [[1, 2, 3], [4, 5, 6]]
    # fig - это сам график 
    # axes - это массив осей
    # функция plt.subplots() - принимает количество строк и столбцов, figsize - это размер графика, принимает кортеж (ширина, высота)
    # Значение  5 * n_rows - это высота графика, 15 - это ширина графика
    fig, axes = plt.subplots(n_rows, n_cols, figsize = (15, 5 * n_rows))
    

    # функция flatten() - преобразует массив осей в одномерный массив
    # axes - представляет из себя массив осей, визуально: 
    # [[1, 2, 3], [4, 5, 6]] -> [1, 2, 3, 4, 5, 6]
    axes = axes.flatten()
    

    # фцнкция enumerate() - возвращает индекс и значение элемента в массиве
    for i, feature in enumerate(num_features):
        try:
            # Работаем уже с одной осью, и конкретным признаком
            ax = axes[i]

            # Исправляем передачу данных в histplot
            # sns - это библиотека seaborn, она предназначена для создания красивых графиков
            # histplot - это функция, которая строит гистограмму
            # data - это датафрейм, в котором наход-ятся данные
            # x - это признак, который будет на гистограмм, то есть он будет распределен по оси x
            # hue - это признак, который будет использоваться для цветовой маркировки
            #
            # kde (Kernel Density Estimation) - это метод оценки плотности распределения данных
            # Когда kde=True, поверх гистограммы будет построена плавная кривая, показывающая
            # приближенную плотность распределения признака. Это помогает лучше визуализировать
            # форму распределения данных, сглаживая "ступенчатость" гистограммы.
            # Кривая строится с помощью усреднения множества нормальных распределений,
            # центрированных на каждом наблюдении
            #
            # ax - это ось, на которой будет построен график, ax - значит что график будет построен на оси ax
            # element = 'step' - это признак, который будет использоваться для построения графика, значение 'step' для построения графика в виде ступеней ещё есть 
            # значения 'bars', 'poly', 'bars' это другие виды графиков
            #
            # palette - это признак, который будет использоваться для цветовой маркировки, значение ['skyblue', 'salmon'] для цветовой маркировки
            sns.histplot(data=dataframe, x=feature, hue='is_real_incident', 
                        kde=True, ax=ax, element='step',
                        palette=['skyblue', 'salmon'])
            

            ax.set_title(f'Распределение признака  {feature}')

            # Стоит ' ' - это означает что на оси x не будет подписей
            ax.set_xlabel('')
        
        # Ошибка np.linalg.LinAlgError - значит что признак не является числом
        except np.linalg.LinAlgError as e:
            print(f"Ошибка при построении графика для признака {feature}: {e}")

            print('Есть признак, в котором значния не являются числами {feature}')      
            print('#################Ошибка {e}#################')

            sns.histplot(data=dataframe, x=feature, hue='is_real_incident', 
                        ax=ax, element='step',
                        palette=['skyblue', 'salmon'])            

            ax.set_title(f'Распределение признака {feature} (KDE skipped)')
            ax.set_xlabel('')
    
    # Уберем те оси, не использовали в построении, они могли появиться 
    # 1) из-за ошибки 
    # 2) из-за того что признак не является числом(но мы такое обработали в exception)
    for i in range(i + 1, len(axes)):
        # Параметр axis('off') - убирает оси
        axes[i].axis('off')

    # Параметр tight_layout() - подгоняет размеры графиков, чтобы они не накладывались друг на друга
    plt.tight_layout()
    plt.show()
    

exclude_from_plots = ['is_real_incident', 'incident_id']
plot_distribution(df, exclude_cols = exclude_from_plots)



#######################_Блок 7_#######################
print("#################_Обнаружение и обработка выбросов_#################")

# Выбросы - это значения признаков, которые сильно отличаются от остальных значений признака
# Они могут сильно влиять на результаты модели
# Например, если в признаке income есть выбросы, то это может сильно исказить результаты модели

def bplot_boxplot(dataframe, n_cols = 3, exclude_cols = None):

    if exclude_cols is None:
        exclude_cols = []

    num_features = [col for col in dataframe.columns 
                    if col not in exclude_cols 
                    and dataframe[col].dtype in [int64, float64]]

    n_rows = int(np.ceil(len(num_features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize = (18, 5 * n_rows))

    axes = axes.flatten()

    for i, feature in enumerate(num_features):

        ax = axes[i]

        # y - это признак, который будет на оси y, то есть он будет распределен по оси y (Отличие от записи: ax = ax В том что ax - это ось, а y - это признак, который будет на оси y), если имеем дело с x, то: 
        # x - это признак, который будет на оси x, то есть он будет распределен по оси x; 
        #
        # palette - это признак, который будет использоваться для цветовой маркировки;
        sns.boxplot(data = dataframe, x = 'is_real_incident', y = feature, ax = ax, palette = ['skyblue', 'salmon'])
        ax.set_title(f'Распределение признака {feature}')
        ax.set_xlabel('is real incident')

    for i in range(i + 1, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# Покажем выбросы через ящики с усами: 
print("Work with BoxPlot")
plot_boxplot(df, exclude_cols = exclude_from_plots)


# Обрабтаем выбросы c использование IQR
def handle_outlier(dataframe, columns):
    
    df_cleaned = dataframe.copy()

    for col in columns:

        # Определеми кваритили
        Q_1 = df_cleaned[col].quantile(0.25)
        Q_3 = df_cleaned[col].quantile(0.75)
    
        IQR = Q_3 - Q_1

        # Опредеоим границы
        lower_bound = Q_1 - 1.5 * IQR 
        upper_bound = Q_3 + 1.5 * IQR

        # Поисчиатем число выбросов 
        outliers_count = (df_cleaned[col] < lower_bound | df_cleaned[col] > upper_bound).sum()
        percent_outliers = (outliers_count / len(df_cleaned)) * 100

        print(f"Признак {column}: обнаружено {outliers_count} выбросов ({outliers_percent:.2f}%)")

        #  метод .clip() - ограничивает значения признака в пределах заданного диапазона
        df_cleaned[col] = df_cleaned[col].clip(lower_bound, upper_bound)

    return df_cleaned

# функция select_dtypes() - выбирает признаки по типу данных, параметр include = ['number'] - выбирает признаки по типу данных число
# методо columns.tolist() - преобразует признак в список, для того чтобы можно было использовать в цикле
numeric_features = df.select_dtypes(include = ['number']).columns.tolist()
cols_for_outlier_detection = [col for col in numeric_features if col not in exclude_from_plots]

# Обработали выбросы с использованием IQR
df_cleaned = handle_outlier(df, cols_for_outlier_detection)


#######################_Блок 8_############################################################################
print("Проведем анализ корреляции признаков")


# метод .figure() - создает фигуру
# figsize - это размер графика, принимает кортеж (ширина, высота)
plt.figure(figsize=(16, 14))




################################__О корреляции__################################
# В переменной corelation_matrix - имеем матрицу.
# Под капотом метод .corr() устроена формула Пирсона
# Формула Пирсона:
# r = (n * sum(x * y) - sum(x) * sum(y)) / sqrt((n * sum(x^2) - sum(x)^2) * (n * sum(y^2) - sum(y)^2))
# n - количество строк
# x - признак 1
# y - признак 2

# Она работает так: 
# 1) Берется признак 1 и признак 2
# 2) Берется среднее значение признака 1 и признака 2
corelation_matrix = df.select_dtypes(include = ['number']).corr()

# Метод из библиотеки np .triu() создает верхний треугольник матрицы из списка сorelation_matrix
mask = np.triu(corelation_matrix)



######################################__Построение тепловой карты__######################################

# Библиотека sns - библиотека seaborn, она предназначена для создания красивых графиков
# Метод heatmap() - создает тепловую карту
# параметр annot для того чтобы были надписи на графике
# Метод fmt - формат чисел на графике, Значение .2f означает что числа будут округлены до 2 знаков после запятой
# Метод cmap - цветовой профиль, Значение coolwarm означает что цвета будут меняться от синего до красного
# Метод mask - маска, Значение mask означает что на графике будут видны только верхний треугольник матрицы, то есть можем на figure наложить маску.
# Метод linewidths - ширина линий на графике, Значение 0.5 означает что линии будут иметь ширину 0.5
# Метод cbar_kws - параметры цветовой шкалы, Значение shrink для того чтобы уменьшить размер цветовой шкалы
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            mask=mask, linewidths=0.5, cbar_kws={"shrink": .8})

plt.title('Корреляционная матрица числовых признаков', fontsize=16)

# Метод xticks() - подписывает оси x, параметр rotation=45, ha='right' - означает что подписи будут повернуты на 45 градусов и будут выровнены по правому краю
plt.xticks(rotation=45, ha='right')

# Метод tight_layout() - подгоняет размеры графиков, чтобы они не накладывались друг на друга
plt.tight_layout()
plt.show()



#######################_Блок 9_#######################
print("#################_Анализ связи признков c целевой переменной_#################")

# То есть посмотрим на то, на сколько коррелируют признаки с целевым признаком

# Методо .sort_values() - сортирует значения признака по убыванию
# ascending=False - означает что сортировка будет по убыванию
target_corelation = corelation_matrix['is_real_incident'].sort_values(ascending=False)

print(target_corelation)

# Покажем это наглядно:
plt.figure(figsize=(12, 8))
sns.barplot(x=target_correlation.index[:10], y=target_correlation.values[:10])
plt.title('Топ-10 признаков по корреляции с is_real_incident')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



#######################_Блок 10_#######################

print("#################_Изучение зависимости между признаками_#################")

# Изучение зависимости между признаками — это более широкий и комплексный подход, который включает анализ любых взаимосвязей 
# (линейных и нелинейных) между переменными с использованием различных методов (визуализация, регрессионный анализ, методы машинного обучения и т.д.). 
# В то же время анализ корреляции — это специализированный метод, 
# ориентированный преимущественно на измерение степени линейной зависимости между двумя признаками с помощью коэффициентов (например, коэффициента Пирсона). 
# Таким образом, корреляционный анализ является лишь одним из инструментов в рамках общего изучения взаимосвязей между признаками.
#


# Что такое scatter plot?
# Scatter plot — это график, который показывает зависимость между двумя признаками.
# Он состоит из точек, которые показывают значения признаков.
def bild_scatter_plot(dataframe, features, target_col = 'is_real_incident'):

    # Создадим подвыборку для быстроты построения 

    if len(dataframe) > 1000:
        sample_dataframe = dataframe.sample(1000, random_state = 42)

    else: 
        sample_dataframe = dataframe

# Создадим pairplot - Это парные диаграммы рассеивания, для того, чтобы отследить каждого признака с каждым
# data=sample_df - для одщих данных, всех данных
# vars=features - для признаков, которые будут использоваться для построения графика
# hue=target_col - 
# diag_kind='kde' - 
# palette=['skyblue', 'salmon'] - 
# plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'k'} - для параметров графика, в данном случае это alpha - прозрачность, s - размер точек, edgecolor - цвет точек
    sns.pairplot(data=sample_df, vars=features, hue=target_col, diag_kind='kde',
                palette=['skyblue', 'salmon'], plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'k'})

    plt.suptitel("Матрица для диаграмм рассеивания признаков")

    plt.tight_layout()
    plt.show()


# Что возвращает метод .index?
# Метод .index возвращает индексы признаков, то есть имена признаков, напрмер:
#  ['is_real_incident', 'incident_id', 'after_order_start', 'after_order_start_day',
#  'after_order_start_month', 'is_weekend', 'incident_severity', 'incident_type', 'incident_subtype', 'incident_category']
#
# Что за синтаксис такой:
# [...][:5]
# Это срез списка, то есть мы берем первые 5 элементов списка
top_features = [col for col in target_corelation.index 
                if col not in exclude_from_plots if col != 'is_real_incident'][:5]

bild_scatter_plot(df_cleaned, top_features)



#######################_Блок 11_#######################
print("#########Трансформация и масштабирование признаков#########")

# Функция для логаримического преобразования признаков с ассиметричным распределением
# Ассиметричное распределение - это когда распределение признака не симметрично, то есть одна сторона распределения тяжелее другой
def log_transform(dataframe, features):

    df_transformed = dataframe.copy()

    for feature in features:

        min_value = df_transformed[feature].min()

        if min_value <= 0:

            # Добавим константу, чтобы все значения были положительными
            shift = abs(min_value) + 1

            df_transformed[feature] = df_transformed[feature] + shift
            df_transformed[feature] = np.log1p(df_transformed[feature])

            print(f"Применено log-преобразование к {feature} с добавлением константы {shift}")
        else: 

            df_transformed[feature] = np.log1p(df_transformed[feature])
            print(f"Применено log-преобразование к {feature}")

    return df_transformed


# Выбрали признаки с асимметричным распределением
skewed_features = ['total_after_distance_km', 'max_jump_distance_km', 'max_speed_kmh',
                  'radius_from_start_m', 'coordinate_stability_m']

print("Применение логарифмического преобразования к признакам с асимметричным распределением:")
df_transformed = log_transform(df_cleaned, skewed_features)



#######################_Блок 12_#######################
print("#########Масштабирование признаков#########")

# Масштабирование признаков - это процесс приведения признаков к одному масштабу
# Это необходимо для того, чтобы модели машинного обучения работали лучше

def scale_features(dataframe, features, type_scaling = 'standard'):

    df_scaled = dataframe.copy()

    if type_scaling == 'standard':
        scaler = StandardScaler()

    elif type_scaling == 'minmax':
        scaler = MinMaxScaler()
        

    # Масштабирование в зааисимости от типа масштабирования
    scale_features = scaler.fit_transform(df_transformed[features])


    #Добавим отмасштабированные признаки в датафрейм
    for i, feature in enumerate(features):

    # Что за синтаксис f'{feature}{suffix}
    # Это форматирование строки, то есть мы берем признак и добавляем к нему suffix

        df_scaled[f'{feature}{suffix}'] = scaled_features[:, i] 

        print(f"Применено {type_scaling} масштабирование к {feature}")

    return df_scaled, scaler

features_to_scale = [col for col in df_transformed.select_dtypes(include=['number']).columns
                    if col not in ['incident_id', 'is_real_incident']]


# Применение StandardScaler
df_scaled, scaler = scale_features(df_transformed, features_to_scale, type_scaling = 'standard')

# Применение MinMaxScaler
df_scaled_minmax, scaler_minmax = scale_features(df_transformed, features_to_scale, type_scaling = 'minmax')



#######################_Блок 13_#######################
print("Выделение признаков с помощью feature importance")


# метод .drop() - удаляет признаки из датафрейма, для того, 
X = df_cleaned.drop(['incident_id', 'is_real_incident', 'last_order_end', 'after_order_start'], axis=1)
y = df_cleaned['is_real_incident']

# Оценим важность каждого признака при помощи SeleсtKBest (функция из библиотеки sklearn)
# Параметр k  означает что мы выберем k признаков с наибольшей важностью
# Параметр score_func = f_classif означает что мы будем использовать F-статистику  для оценки важности признаков
# (F статистика простыми слованми это отношение дисперсии к среднему квадратичному отклонению) 
selector = SelectKBest(score_func=f_classif, k='all')
selector.fi(X, y)


