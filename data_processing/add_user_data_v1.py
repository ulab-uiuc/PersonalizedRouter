import pandas as pd


def expand_dataframe(df):
    expanded_rows = []

    for _, row in df.iterrows():
        for i in range(2, 11):
            row_copy = row.copy()
            row_copy['user_id'] = i - 1
            row_copy['performance_preference'] = round(0.1 * i, 1)
            expanded_rows.append(row_copy)

    df_expanded = pd.DataFrame(expanded_rows)
    column_order = ['user_id', 'performance_preference'] + [col for col in df.columns]
    df_expanded = df_expanded[column_order]
    df_expanded = df_expanded.sort_values(by=['user_id', 'task_id'])
    df_expanded['reward'] = (
            df_expanded['performance_preference'] * df_expanded['effect']
            - (1 - df_expanded['performance_preference']) * df_expanded['cost']
    )

    df_expanded['group_id'] = df_expanded.index // 10

    df_expanded['best_llm'] = df_expanded.groupby(['user_id', 'group_id'])['reward'].transform(
        lambda x: x.idxmax() % 10
    )

    df_expanded = df_expanded.drop(columns=['group_id'])

    df_expanded = df_expanded.sort_values(
        by=['user_id', 'performance_preference', 'task_id'],
        ascending=[True, True, True],
        key=lambda col: col.str.lower() if col.name == 'task_id' else col
    )

    return df_expanded


input_csv_path = "../data/router_data.csv"
output_csv_path = "../data/router_user_data_v1.csv"

df_original = pd.read_csv(input_csv_path)
df_expanded = expand_dataframe(df_original)

df_expanded.to_csv(output_csv_path, index=False)