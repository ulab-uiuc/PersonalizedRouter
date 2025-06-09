import pandas as pd
import numpy as np

router_df = pd.read_csv('../data/router_data_v2.csv')
judge_df = pd.read_csv('../data/llm_judge_results.csv')

answers_per_query = 10
num_queries = len(router_df) // answers_per_query

expanded_rows = []

for query_idx in range(num_queries):
    start_idx = query_idx * answers_per_query
    end_idx = start_idx + answers_per_query
    query_block = router_df.iloc[start_idx:end_idx].copy()
    judgements = judge_df.iloc[query_idx * 9: (query_idx + 1) * 9]

    for _, judge_row in judgements.iterrows():
        judger_idx = judge_row['judger_idx']
        best_idx = int(judge_row['judge_choice'])
        user_id = int(judger_idx) + 1

        temp_block = query_block.copy()
        temp_block['user_id'] = user_id
        temp_block['best_answer'] = [1 if i == best_idx else 0 for i in range(answers_per_query)]

        expanded_rows.append(temp_block)

final_df = pd.concat(expanded_rows, ignore_index=True)

min_cost = final_df['cost'].min()
max_cost = final_df['cost'].max()
final_df['cost'] = (final_df['cost'] - min_cost) / (max_cost - min_cost)

min_effect = final_df['effect'].min()
max_effect = final_df['effect'].max()
final_df['effect'] = (final_df['effect'] - min_effect) / (max_effect - min_effect)

cols = final_df.columns.tolist()
cols.insert(0, cols.pop(cols.index("user_id")))
final_df = final_df[cols]

final_df = final_df.sort_values(
    by=['user_id', 'task_id'],
    ascending=[True, True],
    key=lambda col: col.str.lower() if col.name == 'task_id' else col
)

final_df.to_csv('../data/router_user_data_v2.csv', index=False)
