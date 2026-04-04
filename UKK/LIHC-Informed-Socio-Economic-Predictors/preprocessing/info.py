
# import the reuqired packages (libraries and modules)

import pandas as pd










def dataframe(path):

    

    # main data of EP
    df_main = pd.read_excel(path, sheet_name='ENABLE.EU_dataset_HH')

    # variable info
    df_variables = pd.read_excel(path, sheet_name='variables (questions) labels')

    # variable values
    df_label_value = pd.read_excel(path, sheet_name='variables value (answers) label')


    print("--------- what you need to know about main data -----------------")

    print(df_main.loc[:5])

    print("******************** list of columns ***************")
    col_list = df_main.columns.to_list()

    print(col_list)

    print("******************** columns' info ***************")


    print(df_main.info())

    print("******************** missing values ***************")
    print(df_main.isnull().sum())


    print("******************** statistically description of the data ***************")
    print("count - The number of not-empty values.")
    print("mean - The average (mean) value.")
    print("std - The standard deviation.")
    print("min - the minimum value.")
    print("25% - The 25% percentile.")
    print("50% - The 50% percentile.")
    print("75% - The 75% percentile.")
    print("max - the maximum value.")
    print(df_main.describe(include='all'))


    return df_main, df_variables, df_label_value, col_list

