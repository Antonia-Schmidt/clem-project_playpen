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

### Training Data
Contains the datasets used for training the models found in hugging face.

### _old
Refers to data from benchmark versions 0.9 and 1.0

### _new
Refers to data from benchmark version > 1.0