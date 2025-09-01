# Haiku
A silly bot that (imperfectly) turns any Japanese sentence into a haiku

## 1. Purpose
We want to fine-tune a pretrained small language model (SLM) to convert any Japanese sentence into a haiku. While this task can be done easily with a large language model (LLM), it remains challenging for an SLM (with fewer than 1B parameters). Our goal is to explore whether it's possible to create a haiku generator using an SLM.

The optimization objective of this fine-tuning process is to guide the model to produce output that resembles a haiku as closely as possible. Haiku follows several lexical and semantic rules, but for simplicity, we will only enforce the syllable pattern of each line. This can be checked programmatically, and we will use it as a reward function while leveraging [GRPO](https://huggingface.co/papers/2402.03300) to train our model.

## 2. Package layout
- [`./prompts/`](./prompts/) directory: prompts that will be input to the model.
- [`./data.py`](./data.py):
    - Fetch data from Yahoo Japan ニュース: https://news.yahoo.co.jp/rss
    - Generate training and validation datasets.
- [`./train.py`](./train.py):
    - `SyllableCounter`: a utility class that helps count the number of syllables in a given Japanese or English sentence.
    - `RewardFunctionBuilder`: construct the reward function used with GRPO during model training.
        - `_score_syllable_count` method: reward completions that follow the syllable pattern rule and penalizes those that do not.
        - `_reward_existent_characters` method: reward completions that contain characters from the original sentence to help prevent hallucination.
        - `_penalize_line_overlap` method: penalize completions that repeat the same words across multiple lines.
    - And other functions used for training and validating the model.

## 3. Best practices
I figured out some tips to train GRPO better:

### 3-1. "Smooth" reward function
Design reward functions where scores change "smoothly" and avoid "gaps" across different model completions.

For example, a haiku must have exactly 3 lines. However, you *should not* assign a score of 1 only when the output has exactly 3 lines, and 0 otherwise. GRPO helps models learn through exploration, and early on, there's very little chance of producing exactly 3 lines. A reward function with such a sharp "gap" forces the model to "jump" to a perfect result without any guidance in between.

Instead, create a reward function that assigns scores in a 0-1 range, where outputs closer to 3 lines receive higher scores. This kind of smooth reward encourages gradual improvement and helps the model explore more efficiently.

### 3-2. Fight against hallucination
This is an eternal problem when working with LLMs, and TBH, I have no idea what the best approach is. In this case of training a haiku model, I simply use `_reward_existent_characters` reward function to encourage the model to output characters that exist in the original sentence, hoping it won't try to make things up.

### 3-3. Cover broad cases
AI is like a human: lazy and always looking for the shortest, fastest way to reach a goal. If your reward function isn't robust, AI will find a way to "cheat the rules". To prevent this, there's only one choice: write reward functions, train the model, check if it's cheating, and revise the functions to address it.

For example, during training, the model may start repeating the same words over and over, as long as it sticks to the 5-7-5 syllable pattern of haiku. We had to add `_penalize_line_overlap` method as one of the reward functions to tell the model: "Hey, repeating words is cheating. Be more creative instead of just following Haiku rules".

## 4. Results
After running some tests, I realized that training an SLM in this way is still difficult. The model isn't smart enough to generate a haiku whose lines form a meaningful sentence. Instead, it simply puts words together, lacking a coherent connection between them.

<details><summary>Sample 1 (from evaluation dataset)</summary>

Original sentence:
```
【軽井沢】軽井沢ホテルブレストンコート／個性豊かな会場をスタイルに合わせてセレクトHoshino Resorts Karuizawa Hotel Bleston Court古くから避暑地として愛されてきた軽井沢
```
Pretrained completion:
```
轻轻地走在轻井沢の水道旁，
各個性豊かな会場をZanzibar Styleに選び
に静かに walkers と gapminder ラフピアに荒凉な所を数えて
なる缩写の相俟いが楽しみ-medium medium mediumartz<br>
The complex of the appropriate <br>
rooms at Zanzibar Style are located near the cafe <br>
and beyond the mountain take a right under a few years pass written by Amakura Haruka.
```
Trained LoRA completion:
```
軽井沢ホテル
個性豊かな会場
スタイルセレクト
```
</details>

<details><summary>Sample 2 (from evaluation dataset)</summary>

Original sentence:
```
圧巻のローズ・ボールルック1Getty Imagesモナコ公室が主催するチャリティ・イベント、ローズ・ボールでの装いは常に注目の的
```
Pretrained completion:
```
ローズに色の点に"os brosukita"（おろせる0点にほんとうに）
在جمは式し始め着しルック(urojubi o kanigyotachi)に altındaうし
美しいモナコ公室では光役のカクテルに�
ия努カラヌトスに做了刺のローズ.CENTER.SOCKS ESPALHADA
```
Trained LoRA completion:
```
圧巻ローズ
ボールルック主催
注目装い
```
</details>

<details><summary>Sample 3 (another Japanese sentence)</summary>

Original sentence (copied from https://corp.moneyforward.com/aboutus/mission/):
```
「お金」は、人生においてツールでしかありません。しかし「お金」とは、自身と家族の身を守るため、また夢を実現するために必要不可欠な存在でもあります。私たちは「お金と前向きに向き合い、可能性を広げることができる」サービスを提供することにより、ユーザーの人生を飛躍的に豊かにすることで、より良い社会創りに貢献していきます。
```
Pretrained completion:
```
お金は手にできるツール唯が、持ち歩く echtただしお金は自身を守る当然の存在も人生の Groß potatoes
夢と可能性を広げると一層豊かに(parsed)成長し、新たな可能性を生み出したり
この社会がける.GREEN的社会利子
```
Trained LoRA completion:
```
お金ツール
自身守るため
夢あり社会
```
</details>

<details><summary>Sample 4 (we don't train with English data, but let's see if the model can handle it)</summary>

Original sentence (copied from https://github.com/features/copilot):
```
Deals with your issues. When assigned issues, GitHub Copilot plans, writes, tests, and iterates—using GitHub Actions to run code and deliver ready-to-review pull requests. Codes like an insider. GitHub Copilot hooks into MCP servers to draw on data from your repositories and external resources—working like an onboarded team member from day one. Human and agent in the loop. Comment to guide GitHub Copilot, polish your code for merge, or take over locally in your IDE.
```
Pretrained completion:
```
GitHub Copilot skillfully processes tasks.
At its core, it's a collaborative platform.
Its agents, like the insiders mentioned,
guide you through your code, polish, or refactor.
```
Trained LoRA completion:
```
Issues follow
GitHub Copilot writes
Data uses inside
```
</details>
