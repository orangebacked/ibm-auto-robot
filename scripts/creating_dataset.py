import pandas as pd




df_failed_dev = df[df['device'].isin(list(df[df['failure'] == 1]['device']))]
df_not_failed_dev = df[df['device'].isin(list(df[df['failure'] == 0]['device']))]