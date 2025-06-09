import pandas as pd
from openai import OpenAI
from tqdm import tqdm


client = OpenAI(api_key="",
                base_url='')

df = pd.read_csv("../data/router_data.csv")

judge_prompts = [
    "You are a person full of sensibility and you tend to choose answers that are natural, warm and relatable rather than overly formal or calm expressions.\n",
    "You are an inquisitive young person and you prefer answers that are creative and light-hearted with humor.\n",
    "You are a math enthusiast and you tend to choose answers that are clearly explained, step-by-step, and have a logical process.\n",
    "You are an engineer who prefers answers that are simple and direct, especially those that lead to conclusions through practical calculations and formulae.\n",
    "You are a student and you prefer answers that contain detailed explanations and help you understand the concepts.\n",
    "You are an information retrieval specialist and you tend to choose answers that answer the question precisely and where the answer is highly relevant to the context.\n",
    "You are a news editor who prefers summaries that contain all the important information, are logical, and are concise.\n",
    "You are a literature enthusiast, who tends to prefer answers that are eloquent, rhetorically rich, and capable of conveying deep emotions.",
    "You are an expert in early childhood education, who prefers explanations that use simple language, are vivid and engaging, easy to understand, and inspiring.",
]
append_text = ("Given the Query and 10 answers, you need to select the best answer that you are most satisfied.\n"
               "Ensure that the order of the responses does not influence your decision.\n "
               "Do not let the length of the responses impact your evaluation.\n\n"
               "The system's input is in this format:\n"
               "[User Query]\n"
               "{query}\n"
               "[The Start of Answer 1]\n"
               "{answer_1}\n"
               "[The End of Answer 1]\n"
               "..."
               "[The Start of Answer 10]\n"
               "{answer_10}\n"
               "[The End of Answer 10]\n\n"
               "Your response can only include the answer number, ranging from 1 to 10, no anything else.")

judge_prompts = [f"{prompt} {append_text}" for prompt in judge_prompts]

def make_user_prompt(query, responses):
    formatted_responses = "\n".join([f"[The Start of Answer {i}]\n{resp}\n[The End of Answer {i}]\n" for i, resp in enumerate(responses)])
    prompt = (
        f"[User Query]\n{query}\n"
        f"{formatted_responses}"
    )
    return prompt

results = []
num_rows = df.shape[0]

for i in tqdm(range(0, num_rows, 2)):
    block = df.iloc[i:i+2]
    query = block["query"].iloc[0]
    responses = block["response"].tolist()

    for idx, system_prompt in enumerate(judge_prompts):
        try:
            completion = client.chat.completions.create(
                model="",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": make_user_prompt(query, responses)}
                ],
                temperature=0.7
            )
            reply = completion.choices[0].message.content.strip()
        except Exception as e:
            reply = f"Error: {str(e)}"

        results.append({
            "query": query,
            "judger_idx": idx,
            "judge_choice": reply
        })

results_df = pd.DataFrame(results)
results_df.to_csv("../data/llm_judge_results.csv", index=False)
