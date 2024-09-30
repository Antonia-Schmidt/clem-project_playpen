This is the repository for the CLEM-ProjectPlayPen.

[Huggingface Page to Check the Models](https://huggingface.co/Nicohst)


# Data

## Structure

### Extracted Information:
The following columns are extracted for each game:

| Entries              | Descriptions                                                                                                                                                               |
|---------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| game                 | The Game e.g. Taboo, Wordle                                                                                                                                                |
| benchmark_version    | Version of the benchmark the data comes from                                                                                                                               |
| game_id              | Unique identifier for the game instance                                                                                                                                    |
| model                | The Model Name                                                                                                                                                             |
| experiment           | The version of the game e.g. 0_high_en (Taboo) or 1_random_grids  (Imagegame)                                                                                              |
| episode              | The Played Episode                                                                                                                                                         |
| Aborted              | 1 if episode was aborted                                                                                                                                                   |
| Lose                 | 1 if episode was played but ended in loss                                                                                                                                  |
| Success              | 1 if episode ended as a success                                                                                                                                            |
| target               | The objective of the game. e.g. target_word (Taboo), target_grid (Imagegame)                                                                                               |
| chat                 | The whole conversation in the huggingface chat format                                                                                                                      |
| player               | The player that was playing. If a game has two players, there will be the exact same instance twiche but on time with the chat of player 1 and once with chat of player 2. |

A game instance can be identified by combining game-benchmark_version-model-experiment-episode.
If a game has 2 players, the same identifier exists twice. One entry represents the conversation of player 1 and the other one represents
the conversation of player 2.

NOTE: This format only applies to all data in the processed folder. The raw data comes in a similar format but with other game
specific columns that were combined into chat or target to unify the data.

### Raw
Contains all data extracted from the clembench-runs repository without any preprocessing

### Processed
Contains the same data as raw, but in a unified format such that all games can be joined together in one file to create datasets.

NOTE: Image game has been preprocessed to mach the format of the newer benchmark version.
In particular, only the answers of player one were parsed to another format. The content however was not changed.

Further, All games were filtered such that there are no chats that contain any INVALID_FORMAT strings.
This string is set as a placeholder in some games where the model is asked to provide an explanation for the answer.
If there is no explanation, there is a INVALID_FORMAT string. To avoid learning this from the data, episodes where this happened were discarded.

### Training Data
Contains the datasets used for training the models found in hugging face.

The naming of the datasets follows the pattern: <br>
D[E][XXXX] where E referrs to the experiment number and XXXX is the sub experiment run within an experiment.<br>

E.g: D10001, is a dataset of experiment 1 and refers to the sub-experiment-0001

### _old
Refers to data from benchmark versions 0.9 and 1.0

### _new
Refers to data from benchmark version > 1.0

# Experiments
### 1. Train on all Successful episodes of all models
**Dataset D10001:** [training_data_D10001.csv](./data/training_data/D10001.csv)

### 2. Train on all Successful episodes of the top n models
Top 10 models (most successful episodes) in the benchmark versions 0.9 and 1.0 combined:

| Place | Item |
|-------|------|
| 1 | gpt-4-0613-t0.0--gpt-4-0613-t0.0 |
| 2 | claude-v1.3-t0.0--claude-v1.3-t0.0 |
| 3 | gpt-4-1106-preview-t0.0--gpt-4-1106-preview-t0.0 |
| 4 | gpt-4-t0.0--gpt-4-t0.0 |
| 5 | gpt-4-0314-t0.0--gpt-4-0314-t0.0 |
| 6 | claude-2.1-t0.0--claude-2.1-t0.0 |
| 7 | gpt-4-t0.0--gpt-3.5-turbo-t0.0 |
| 8 | claude-2-t0.0--claude-2-t0.0 |
| 9 | gpt-3.5-turbo-1106-t0.0--gpt-3.5-turbo-1106-t0.0 |
| 10 | gpt-3.5-turbo-0613-t0.0--gpt-3.5-turbo-0613-t0.0 |

**Dataset n=10 D20001:** [training_data_D20001.csv](./data/training_data/D20001.csv) </br>
**Dataset n=3 D20002:** [training_data_D20002.csv](./data/training_data/D20002.csv) </br>
**Dataset n=1 D20003:**  [training_data_D20003.csv](./data/training_data/D20003.csv) </br>

### 3. Train on partial conversation pieces
The previous experiments all take a whole interaction for an episode and provides only the final answer to learn for the model.
This means that intermediate turns are not to be learned by the model. This experiment takes the whole interaction and splits it into n pieces.

Each piece is on part of the interactions. The last piece contains the whole conversation while the first only contains one instruction and one answer.

Since in some games, the conversation starts with two user inputs before the fist assistant answer, the two consecutive user inputs will be merged into one 
using clench utils functionality. This functionality is in [utils/utils.py](./src/utils/utils.py)

The experimets are the following:</br>
**Dataset D30001:** [training_data_D30001.csv](./data/training_data/D30001.csv)  Contains the conversation pieces of all successful episodes of all models equivalent to D10001 without split conversations</br>
**Dataset D30002:** [training_data_D30001.csv](./data/training_data/D30002.csv)  Contains the conversation pieces of all successful episodes of the top 10 models equivalent to D20001 without split conversations</br>
**Dataset D30003:** [training_data_D30001.csv](./data/training_data/D30003.csv)  Contains the conversation pieces of all successful episodes of the top 3 models equivalent to D20002 without split conversations</br>
**Dataset D30004:** [training_data_D30001.csv](./data/training_data/D30004.csv)  Contains the conversation pieces of all successful episodes of the top 1 equivalent to D20003 without split conversations</br>

NOTE: The data was shuffled before it was split to mix the games. During training the data was not further split to keep the order of the conversatoin bits.

### 4. Balance the data:
Currently, all data was used regardless of duplicate entries of game instances. This means, that if all models succeeded in on game, then
the same game instance, e.g. wordle with the same target word will be present as often as there are models in the benchmark and version.

<img src="./Plots/Barchart_Top_k_Models_v0_9-1_0.png" alt="Stacked Bar Plot of Successful Episodes">

Since some games are "easier" to play by the models than others, there is the situation, that for some games there are hundreds of samples, while for 
others there are just ten to twenty.

The aboce image already indicates that there even in the top 10 models, some games are only Successfully played by the top 5 a.g. privateshared.

To address this issue there can be different strategies of sampling only partial data from the games that are over represented.

1. Random Sampling
2. Only Top k modes: This sampling tries to take as many episodes from one model as possible and of the model does not have any successful episodes from 
one game episode, the next worse model will be checked.

### 5. Only take data from the best model per game instance (can be duplicate with top n = 1)

### 6. Play only n games and observe impact on other games
(Find reasonable choice of games to learn from)


# Model Naming
The model naming follows the pattern:<br>
[model_name]-[Learning Technique]-E[number of episodes]-[Dataset Identifier] <br>
**model name** is the name of the base-model<br>
**Learning Technique** SFT for supervised fine tuning DPO for direct preference optimization<br>
**E** is the number of trained episodes<br>
**Dataset Identifier** is the dataset id that was introduced in the previous section<br><br>
E.g. meta-llama-Meta-Llama-3.1-8B-Instruct-SFT-E1-D10001
