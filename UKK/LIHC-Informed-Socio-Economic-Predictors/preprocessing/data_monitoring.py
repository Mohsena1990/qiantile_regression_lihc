




import info
import missing_values as missing_values













# path directory
path = r".\ENABLE.EU_dataset_survey of households.xlsx"
df, df_var_description, df_var_values, col_list = info.dataframe(path)



# print(df)
# print(df_var_description)
# print(df_var_values)

# print(handling_missing_value.missing_values(df))

print(missing_values.report_missing_and_unknowns(df))






