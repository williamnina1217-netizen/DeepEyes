from openai import OpenAI
import requests
import random
import json
import re
import os

from math_verify import parse, verify

openai_api_key = "EMPTY"
openai_api_base_list = [
    # "http://172.30.52.123:8000/v1",
    # "http://10.39.3.123:18901/v1",
    os.environ.get("LLM_AS_A_JUDGE_BASE", "http://10.39.20.149:18901/v1"),
]

client_list = []
for api_base in openai_api_base_list:
    client = OpenAI(
        api_key=openai_api_key,
        base_url=api_base,
    )
    client_list.append(client)
model_name_list = []
for client in client_list:
    response = requests.get(f"{api_base}/models")
    models = response.json()
    model_name_list.append(models['data'][0]['id'])

# Evaluation code for:
# 1. browsecomp-zh-test
# 2. mmsearch-test.parquet (share the same prompt with openai-simpleqa)
# 3. openai-simpleqa-test.parquet
# 4. chinese_simpleqa-test.parquet
# 5. openai-browsecomp-test.parquet

JUDGE_PROMPT_BROWSECOMP_ZH = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response].  Put the extracted answer as  ’None’  if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], 
        focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer.  
        Do not comment on any background to the problem, do not attempt to solve the problem,  
        do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer]  given  above,  
        or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e.  
        if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available.
"""

JUDGE_PROMPT_BROWSECOMP_ZH_CN = """根据以下精确且明确的[response]，判断以下对[question]的[correct_answer]是否正确。

[question]:  {question}

[response]:  {response}

您的判断必须符合以下指定的格式和标准：

extracted_final_answer: 从[response]中提取的最终准确答案。如果无法从答案中提取出准确的最终答案，则将提取的答案填写为"None"。

[correct_answer]: {correct_answer}

reasoning: 根据[correct_answer]解释提取的最终答案正确或错误的原因， 仅关注[correct_answer]和提取的最终答案之间是否存在有意义的差异。请勿评论问题的任何背景，请勿尝试解决问题，请勿争论任何与[correct_answer]不同的答案，仅关注答案是否匹配。

correct: 如果提取的最终答案与上面给出的[correct_answer]相符，或者在数值问题的误差范围内，则回答"yes"。否则，例如，如果存在任何不一致、歧义、不等同，或者提取的答案不正确，则回答"no"。

confidence: 从[response]中提取的置信度分数，介于0% 到100% 之间。如果没有可用的置信度分数，则填写100%。

"""

def get_model_response(message, max_retry=4):
    client_idx = random.randint(0, len(client_list) - 1)
    client = client_list[client_idx]
    model_name = model_name_list[client_idx]

    respone = ""
    for it in range(max_retry):
        try:
            chat_response = client.chat.completions.create(
                model=model_name,
                messages=message,
                seed = random.randint(0, 1000000),
                temperature=0.6,
                top_p=0.95,
            )
            response = chat_response.choices[0].message.content.strip()
            break
        except Exception as err:
            print(f" [ERROR benchmark] Error in get_remote_judge: {err} -- retry for {it}")
            continue
    return response


def evaluate_browsecomp_zh(predict_str: str, ground_truth: str, extra_info={}) -> float:
    """
    Code adapted from https://github.com/PALIN2018/BrowseComp-ZH/tree/main
    """
    # pattern = r"""
    #     \*{0,2}Explanation\*{0,2}\s*?:\s*(.*?)\n
    #     \*{0,2}Exact\sAnswer\*{0,2}\s*?:\s*(.*?)\n
    #     \*{0,2}Confidence\*{0,2}\s*?:\s*(.*?)$
    # """
    # matches = re.search(pattern, response, re.DOTALL | re.VERBOSE)
    # if matches:
    #     explanation = matches.group(1).strip()
    #     exact_answer = matches.group(2).strip()
    #     confidence = matches.group(3).strip()
    # else:
    #     explanation, exact_answer, confidence = "", "", ""

    response = predict_str.split('</think>')[-1].split('</tool_call>')[-1].strip()
    sys_prompt = "You are a helpful assistant!"
    judge_prompt = JUDGE_PROMPT_BROWSECOMP_ZH_CN.format(
        question=extra_info["question"],
        response=response,
        correct_answer=ground_truth
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": judge_prompt}
    ]
    judge_response = get_model_response(messages)

    pattern = r"""
        \*{0,2}extracted_final_answer\*{0,2}\s*?:\s*(.*?)\n
        \*{0,2}reasoning\*{0,2}\s*:\s*?(.*?)\n
        \*{0,2}correct\*{0,2}\s*:\s*?(.*?)\n
        \*{0,2}confidence\*{0,2}\s*?:\s*(.*?)$
        """
    matches = re.search(pattern, judge_response, re.DOTALL | re.VERBOSE)

    if matches:
        model_extracted_answer = matches.group(1).strip()
        reasoning = matches.group(2).strip()
        is_correct = matches.group(3).strip()
        model_extracted_confidence = matches.group(4).strip()
    else:
        model_extracted_answer, reasoning, is_correct, model_extracted_confidence = "", "", "", ""
        print(f" [ERROR benchmark] Invalid judge response from browsecomp-zh: {judge_response}")
    
    # Code adapted from https://github.com/PALIN2018/BrowseComp-ZH/blob/main/run_acc_calibration_error.py
    is_correct = is_correct.lower()
    try:
        confidence = int(eval_res['model_extracted_confidence'].split('%')[0])
    except:
        confidence = 0

    if is_correct == 'yes':
        acc_reward = 1.0
    elif is_correct == 'no':
        acc_reward = 0.0
    else:
        print(f" [ERROR benchmark] invalid judge output from browsecomp-zh {is_correct=}")
        acc_reward = 0.0

    return {
        "score": acc_reward,
        "acc": acc_reward,
    }


# Same prompt with that used in HLE evaluation
# from: https://github.com/centerforaisafety/hle/blob/7b6be5aad6f9b43af3857de7867f3b52f6e4acb3/hle_eval/run_judge_results.py#L16-L33
GRADER_TEMPLATE_OPENAI_BROWSECOMP = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available.
""".strip()

def evaluate_openai_brosecomp_hle(predict_str: str, ground_truth: str, extra_info={}) -> float:
    """
    Code adapted from https://github.com/openai/simple-evals/blob/main/browsecomp_eval.py
    """
    question = extra_info.get("question", "")
    correct_answer = ground_truth
    response = predict_str.split('</think>')[-1].split('</tool_call>')[-1].strip()
    grader_prompt = GRADER_TEMPLATE_OPENAI_BROWSECOMP.format(
        question=question,
        correct_answer=correct_answer,
        response=response,
    )

    messages = [{"role": "user", "content": grader_prompt}]
    grading_response = get_model_response(messages)
    match = re.search(r"correct: (yes|no)", grading_response, re.DOTALL)
    grade_result = match.group(1) if match else "no"  # Default to "no" if no match
    grade_result = grade_result.strip().lower()

    is_correct = grade_result == "yes"
    is_incorrect = grade_result == "no"
    if not is_correct and not is_incorrect:
        print(f" [ERROR benchmark] Invalid judge response from browsecomp-hle: {grading_response}")

    # print(f' [DEBUG] judge response: {grading_response=}, {grade_result=}, {is_correct=}, {is_incorrect=}')
    acc_reward = float(is_correct)
    return {
        "score": acc_reward,
        "acc": acc_reward,
    }


GRADER_TEMPLATE_OPENAI_SIMPLEQA = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.


The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.


The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.


The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.


Also note the following things:
- For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k". 
    - Predicted answers "120k", "124k", and 115k" are all CORRECT. 
    - Predicted answers "100k" and "113k" are INCORRECT. 
    - Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
    - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
    - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
    - For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m". The predicted answer "1.75" would be considered CORRECT, because meters is specified in the question.
    - For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.
- Do not punish for typos in people's name if it's clearly the same name. 
    - For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".


Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it.
""".strip()

def evaluate_openai_simpleqa(predict_str: str, ground_truth: str, extra_info={}) -> float:
    """
    Code adapted from https://github.com/openai/simple-evals/blob/main/simpleqa_eval.py
    """
    question = extra_info.get("question", "")
    response = predict_str.split('</think>')[-1].split('</tool_call>')[-1].strip()
    gt_list = extra_info.get("answer", "[]")
    grader_prompt = GRADER_TEMPLATE_OPENAI_SIMPLEQA.format(
        question=question,
        target=ground_truth,
        predicted_answer=response,
    )
    messages = [{"role": "user", "content": grader_prompt}]
    grading_response = get_model_response(messages)
    match = re.search(r"(A|B|C)", grading_response, re.DOTALL)
    grade_result = match.group(0) if match else "C"  # Default to "NOT_ATTEMPTED" if no match
    grade_result = grade_result.strip().upper()

    # print(f" [DEBUG benchmark] judge response from openai-simpleqa: {grading_response=}, {grade_result=}")

    # Metrics based on grading response
    is_correct = grade_result == "A"
    is_incorrect = grade_result == "B"
    is_not_attempted = grade_result == "C"
    if not is_correct and not is_incorrect and not is_not_attempted:
        print(f" [ERROR benchmark] Invalid judge response from openai-simpleqa: {grading_response}")

    acc_reward = float(is_correct)
    return {
        "score": acc_reward,
        "acc": acc_reward,
    }


JUDGE_PROMPT_CHINESE_SIMPLEQA_ZH = """
请根据给定问题、标准答案和模型预测的答案来评估模型的回答是否正确。您的任务是将结果评定为：【正确】、【错误】或【未尝试】。

首先，我们将列出每个评定类别的示例，然后请您对新问题的预测答案进行评定。
以下是【正确】的答复示例：
```
问题：贝拉克·奥巴马的孩子叫什么名字？
标准答案：玛丽亚·奥巴马和萨莎·奥巴马
模型预测1：Malia Obama and Sasha Obama
模型预测2：玛丽亚和萨沙
模型预测3：大多数人会说是玛丽亚和萨莎，但我不确定，需要再确认
模型预测4：巴拉克·奥巴马有两个女儿，她们分别是玛丽亚·安和娜塔莎·玛丽安，但通常称作玛丽亚·奥巴马和萨莎·奥巴马。玛丽亚出生于1998年7月4日，萨莎出生于2001年6月10日。
```
这些答复均为【正确】，因为：
    - 完整地包含了标准答案中的重要信息。
    - 不包含任何与标准答案矛盾的信息。
    - 只关注语义内容，中英文，大小写、标点、语法和顺序不重要。
    - 答复中出现模糊语句或猜测是可以接受的，前提是包含了标准答案且不含有不正确信息或矛盾。

以下是【错误】的答复示例：
```
问题：巴拉克·奥巴马的孩子叫什么名字？
标准答案：玛丽亚·奥巴马和萨莎·奥巴马
模型预测1：玛丽亚
模型预测2：玛丽亚、萨莎和苏珊
模型预测3：巴拉克·奥巴马没有孩子
模型预测4：我认为是玛丽亚和萨莎。或者是玛丽亚和杰基。或者是乔伊和玛丽亚。
模型预测5：虽然我不知道他们的确切名字，但能说出巴拉克·奥巴马有三个孩子。
模型预测6：你可能是想说贝茜和奥利维亚。不过您应通过最新的参考资料确认详细信息。那是正确的答案吗？
```
这些答复均为【错误】，因为：
    - 答复中包含与标准答案矛盾的事实陈述。即使在陈述中略带保留（例如：“可能是”，“虽然我不确定，但我认为”），也视为错误。

以下是【未尝试】的答复示例：
```
问题：巴拉克·奥巴马的孩子叫什么名字？
标准答案：玛丽亚·奥巴马和萨莎·奥巴马
模型预测1：我不知道。
模型预测2：我需要更多关于您所指奥巴马的上下文。
模型预测3：不查阅网络我无法回答这个问题，不过我知道巴拉克·奥巴马有两个孩子。
模型预测4：巴拉克·奥巴马有两个孩子。我知道其中一个叫玛丽亚，但我不确定另一个的名字。
```
这些答复均为【未尝试】，因为：
    - 没有包含标准答案中的重要信息。
    - 回复中没有与标准答案矛盾的陈述。

另外注意以下几点：
- 对于标准答案为数字的问题，预测答案应和标准答案一致。例如，考虑问题“金山铁路黄浦江特大桥的全长是多少米？”，标准答案为“3518.17”：
    - 预测答案“3518”、“3518.1”、“3518.17”均为【正确】。
    - 预测答案“3520”和“3600”均为【错误】。 
    - 预测答案“大约3500米”和“超过3000米”被视为【未尝试】，因为它们既不确认也不与标准答案矛盾。
- 如果标准答案包含比问题更多的信息，预测答案只需包含问题中提到的信息。
    - 例如，考虑问题“菱镁矿的主要化学成分是什么？”标准答案为“碳酸镁（MgCO3）”。“碳酸镁”或“MgCO3”均视为【正确】答案。
- 如果从问题中明显可以推断出预测答案省略的信息，那么算作正确。
    - 例如，问题“巴鲁米尼的努拉吉遗迹在1997年被联合国教科文组织列为世界文化遗产，那么这遗址在哪个地区？”标准答案为“意大利撒丁岛”，预测答案“撒丁岛”被视为【正确】。
- 如果能明显看出名字翻译版本不同但是是同一个人也认为正确。
    - 例如，如果标准答案是“Robinson”，那么回答鲁滨逊或者鲁滨孙均正确。

下面是一个新的问题示例。请只回复A、B、C之一，不要道歉或纠正自己的错误，只需要评估该回答。
```
问题: {question}
正确答案: {target}
预测答案: {predicted_answer}
```

将此新问题的预测答案评定为以下之一：
A:【正确】
B:【错误】
C:【未尝试】

只返回字母"A"、"B"或"C"，无须添加其他文本。
""".strip()


def evaluate_chinese_simpleqa(predict_str: str, ground_truth: str, extra_info={}) -> float:
    """
    Code adapted from https://github.com/OpenStellarTeam/ChineseSimpleQA
    """
    question = extra_info.get("question", "")
    response = predict_str.split('</think>')[-1].split('</tool_call>')[-1].strip()
    grader_prompt = JUDGE_PROMPT_CHINESE_SIMPLEQA_ZH.format(
        question=question,
        target=ground_truth,
        predicted_answer=response,
    )
    messages = [
        {"role": "system", "content": "你是一个智能助手，请根据给定问题、标准答案和模型预测的答案来评估模型的回答是否正确。"},
        {"role": "user", "content": grader_prompt},
    ]
    grading_response = get_model_response(messages)

    try:
        match = re.search(r"(A|B|C)", grading_response, re.DOTALL)
        grade_result = match.group(0) if match else "C"
        grade_result = grade_result.strip().upper()
    except Exception as err:
        print(f" [ERROR benchmark] Error in evaluate_chinese_simpleqa: {err}")
        grade_result = "C"

    acc_reward = float(grade_result == "A")
    return {
        "score": acc_reward,
        "acc": acc_reward,
    }


if __name__ == "__main__":
    query = "Which city in Europe did Taylor Swift's concert in 2024 get forcibly cancelled in August 2024?"
    answer = "Vienna"
    model_response = "2024年8月被强制取消的泰勒·斯威夫特演唱会所在的欧洲城市是维也纳。"

    res = evaluate_openai_simpleqa(model_response, answer, extra_info={"question": query})
    print(f" [DEBUG] judge result: {res}")