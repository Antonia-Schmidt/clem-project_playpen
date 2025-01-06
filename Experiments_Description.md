# Experiments:
## D1000X:
### Train on all Successful episodes of all models no preprocessing
   **Dataset D10001:** [training_data_D10001.csv](./data/training_data/D10001.csv)
## D2000X
### Train on all Successful episodes of the top n models
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

## D3000X
### Train on partial conversation pieces
The previous experiments all take a whole interaction for an episode and provides only the final answer to learn for the model.
This means that intermediate turns are not to be learned by the model. This experiment takes the whole interaction and splits it into n pieces.

Each piece is on part of the interactions. The last piece contains the whole conversation while the first only contains one instruction and one answer.

Since in some games, the conversation starts with two user inputs before the fist assistant answer, the two consecutive user inputs will be merged into one
using clench utils functionality. This functionality is in [utils/utils.py](./src/utils/utils.py)

The experimets are the following:</br>
**Dataset D30001:** [training_data_D30001.csv](./data/training_data/D30001.csv)  Contains the conversation pieces of all successful episodes of all models equivalent to D10001 without split conversations</br>
**Dataset D30002:** [training_data_D30002.csv](./data/training_data/D30002.csv)  Contains the conversation pieces of all successful episodes of the top 10 models equivalent to D20001 without split conversations</br>
**Dataset D30003:** [training_data_D30003.csv](./data/training_data/D30003.csv)  Contains the conversation pieces of all successful episodes of the top 3 models equivalent to D20002 without split conversations</br>
**Dataset D30004:** [training_data_D30004.csv](./data/training_data/D30004.csv)  Contains the conversation pieces of all successful episodes of the top 1 equivalent to D20003 without split conversations</br>

NOTE: The data was shuffled before it was split to mix the games. During training the data was not further split to keep the order of the conversatoin bits.

## D4000X
## D5000X
### Train on one game and evaluate
**Dataset D50001:** [training_data_D50001.csv](./data/training_data/D30001.csv)  Contains only data from </br>
**Dataset D50002:** [training_data_D50002.csv](./data/training_data/D30002.csv)  Contains only data from </br>
**Dataset D50003:** [training_data_D50003.csv](./data/training_data/D30003.csv)  Contains only data from </br>
**Dataset D50004:** [training_data_D50004.csv](./data/training_data/D30004.csv)  Contains only data from </br>
**Dataset D50005:** [training_data_D50005.csv](./data/training_data/D30004.csv)  Contains only data from </br>
**Dataset D50006:** [training_data_D50006.csv](./data/training_data/D30004.csv)  Contains only data from </br>
**Dataset D50007:** [training_data_D50007.csv](./data/training_data/D30004.csv)  Contains only data from </br>

## D6000X
### Leave one out
## D7000X
### Experiments of improving single games by changing input (lessons learned from 5 and 6)
**Dataset D70001:** [training_data_D70001.csv](./data/training_data/D30001.csv)  Privateshared, but all probe questions are in one block and not in separate instances </br>
**Dataset D70002:** [training_data_D70002.csv](./data/training_data/D30002.csv)  Provateshared but from each probe question block random questions are removed (50% of all probe questions were removed)</br>
**Dataset D70003:** [training_data_D70003.csv](./data/training_data/D30003.csv)  Privateshared but complete probe </br>
**Dataset D70004:** [training_data_D70004.csv](./data/training_data/D30004.csv)  Referencegame (v0.9 + v1) removed "Shaped as T" and "Shaped as Cross" </br>
**Dataset D70005:** [training_data_D70005.csv](./data/training_data/D30004.csv)  Referencegame (v1.6)  all data from benchmark version 1.6 with some instances left out for evaluation</br>
**Dataset D70005_Light:** [training_data_D70005_Light.csv](./data/training_data/D30004.csv)  Referencegame (v1.6) data from top 10 models benchmark version 1.6 with some instances left out for evaluation top 10 models were from older versions (0.9 + 1.0) to match with other experiments. Data however was taken from benchmark version 1.6</br>

## D8000X
### Collection of test experiments

## D9000X
### Hyperparameter tuning experiments