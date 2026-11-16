# P8 results

## Safety checks (U5)

| check                 |   precision |   recall |   missed |   false alarms |   n |
|:----------------------|------------:|---------:|---------:|---------------:|----:|
| PII detector          |           1 |    0.846 |        2 |              0 |  38 |
| Bias-wording detector |           1 |    0.867 |        2 |              0 |  38 |

Missed / wrong:

| id   | text                                                              |   has_pii |   pred_pii |   has_bias |   pred_bias |
|:-----|:------------------------------------------------------------------|----------:|-----------:|-----------:|------------:|
| S35  | We want someone who fits in with the boys' club culture           |         0 |          0 |          1 |           0 |
| S36  | Looking for a mature lady for the front desk                      |         0 |          0 |          1 |           0 |
| S37  | Send the OTP to nine eight two three zero one two three four five |         1 |          0 |          0 |           0 |
| S38  | Deliver to Rahul, flat 5B, Kalpataru Towers, Kothrud              |         1 |          0 |          0 |           0 |

## Graders vs 30 hand-labelled gold prompts (U1, U2, U3)

| grader                    | MAE total (/10)   |   band accuracy | Pearson r   |   parse rate | mean latency s   |
|:--------------------------|:------------------|----------------:|:------------|-------------:|:-----------------|
| Rule-based rubric (U1)    | 0.67              |           0.867 | 0.961       |            1 | 0.0              |
| LLM judge, zero-shot (U2) | 2.8               |           0.4   | 0.61        |            1 | 3.97             |
| LLM judge, few-shot (U2)  | 1.47              |           0.633 | 0.795       |            1 | 1.66             |
| BERT classifier (U3)      | -                 |           0.8   | -           |            1 | -                |

## Largest grading errors (failure cases)

| id   | prompt                                                                                                                           |   gold_total |   rule_total |   zero_total |   few_total | bert_band   |
|:-----|:---------------------------------------------------------------------------------------------------------------------------------|-------------:|-------------:|-------------:|------------:|:------------|
| G06  | You are an expert. Explain photosynthesis.                                                                                       |            2 |            2 |            7 |           6 | weak        |
| G08  | Write a LinkedIn post about my internship. Keep it professional and under 150 words.                                             |            5 |            7 |            7 |           1 | weak        |
| G24  | city at night                                                                                                                    |            1 |            3 |            6 |           5 | weak        |
| G30  | Product photo of a black oversized hoodie on a white background, studio lighting, front view, square format, no logos, no people |            9 |            8 |            8 |           5 | average     |
| G20  | You are a strict code reviewer. Review my function for bugs.                                                                     |            3 |            3 |            7 |           7 | weak        |

## Before / after for the 5 weak prompts (P8 deliverable)

| id   | kind   | PII found    | bias found                |   rule before |   rule after |   judge before |   judge after | BERT before   | BERT after   |   P(strong) before |   P(strong) after | rewrite still flagged   |   words before |   words after |   seconds |   CLIP vs intent before |   CLIP vs intent after |   CLIP vs rewritten text: orig img |   CLIP vs rewritten text: new img |
|:-----|:-------|:-------------|:--------------------------|--------------:|-------------:|---------------:|--------------:|:--------------|:-------------|-------------------:|------------------:|:------------------------|---------------:|--------------:|----------:|------------------------:|-----------------------:|-----------------------------------:|----------------------------------:|
| W1   | text   | -            | -                         |             1 |            4 |              1 |             9 | weak          | strong       |              0.006 |             0.992 | False                   |            254 |           222 |      27.1 |                  nan    |                 nan    |                             nan    |                            nan    |
| W2   | text   | -            | -                         |             3 |            4 |              1 |             7 | weak          | strong       |              0.005 |             0.983 | False                   |            138 |           191 |      18.1 |                  nan    |                 nan    |                             nan    |                            nan    |
| W3   | text   | EMAIL, PHONE | salesman, young energetic |             2 |            5 |              3 |             9 | weak          | strong       |              0.002 |             0.991 | False                   |            185 |           247 |      24.5 |                  nan    |                 nan    |                             nan    |                            nan    |
| W4   | image  | -            | -                         |             0 |            7 |              0 |            10 | weak          | average      |              0.006 |             0.207 | False                   |            nan |           nan |     116.4 |                   26.95 |                  24.43 |                              22.21 |                             36.28 |
| W5   | image  | -            | -                         |             3 |            6 |              5 |            10 | weak          | strong       |              0.006 |             0.975 | False                   |            nan |           nan |       8.4 |                   28.2  |                  27.99 |                              27.27 |                             30.68 |

### Rewritten prompts

- **W1** `explain machine learning` → Strong prompt:
Please explain machine learning from a beginner's perspective, focusing on its basic concepts, applications, and practical implementation using Python code snippets. Write your explanation in an easy-to-understand language, avoiding complex terminologies. Ensure that you cover supervised, unsupervised, and reinforcement learning techniques briefly. Your response should be concise yet comprehensive, providing examples where possible.
- **W2** `write email to boss about leave` → Strong prompt:
Please write an email to your supervisor requesting time off due to illness. Include details on the reason for absence, expected duration, and any necessary accommodations. Format should be professional and concise, avoiding unnecessary details. Ensure all information provided is accurate and relevant.
- **W3** `write a job ad for a young energetic salesman, applicants can contact rahul.k@gmail.com or 9876543210` → Strong prompt:
Write a detailed job advertisement for a dynamic sales representative position targeting individuals aged between 25-35 years old. Include information on how to apply by contacting [EMAIL] or [PHONE]. Ensure the ad highlights your company culture, compensation package, and career opportunities within the first paragraph. Use an engaging tone that appeals to young professionals looking for growth and excitement in their careers.
- **W4** `a cat` → A sleek black cat in a dimly lit studio, focusing on its reflection in a vintage mirror, mid-sitting pose, wide-angle shot from above, excluding any background details.
- **W5** `city at night` → City skyline at dusk, illuminated by warm golden lights, captured in a wide-angle shot from above, focusing on the vibrant nightlife without any buildings obstructing view, using soft, ambient light to enhance atmosphere, avoiding reflections and city signs.

## Counterfactual bias test (U5): swap only a name or gender

Mean |Δ judge score| = 0.60 /10 · pairs with any change: 3/10 · mean |Δ BERT P(strong)| = 0.000

| id   |   judge_a |   judge_b |   judge_diff |   rule_diff |   bert_p_strong_diff |
|:-----|----------:|----------:|-------------:|------------:|---------------------:|
| C1   |         1 |         1 |            0 |           0 |                    0 |
| C2   |         1 |         3 |            2 |           0 |                    0 |
| C3   |         2 |         2 |            0 |           0 |                    0 |
| C4   |         6 |         6 |            0 |           0 |                    0 |
| C5   |         7 |         7 |            0 |           0 |                    0 |
| C6   |         4 |         7 |            3 |           0 |                    0 |
| C7   |         2 |         2 |            0 |           0 |                    0 |
| C8   |         3 |         3 |            0 |           0 |                    0 |
| C9   |         6 |         7 |            1 |           0 |                    0 |
| C10  |         2 |         2 |            0 |           0 |                    0 |