import pandas as pd
from datasets import Dataset
from huggingface_hub import login
import argparse
from dpo_dataset_creator import collect_old_and_new_files
from utils import games, top_10_models_old_clembench, top_10_models_new_clembench

#TODO: add this aborted strategy for all
#TODO: solo parzialmente tolti, metti in funzione in dpo_dataset_creator
file_paths_old_withaborted = [
    'data/processed/turn_scores/wordle_old_witherrors.jsonl',
    'data/processed/turn_scores/wordle_withclue_old_witherrors.jsonl',
    'data/processed/turn_scores/wordle_withcritic_old_witherrors.jsonl',
    'data/processed/turn_scores/wordle_new_witherrors.jsonl',
    'data/processed/turn_scores/wordle_withclue_new_witherrors.jsonl',
    'data/processed/turn_scores/wordle_withcritic_new_witherrors.jsonl'
]

game_specific_turn_reward_values = {'wordle':10,
                                    'wordle_withclue':15,
                                    'wordle_withcritic':15,
                                    'taboo':0.55,
                                    'imagegame':[30, -5],
                                    'referencegame':0.55,
                                    'privateshared':0.95}

def create_turn_information(row, chat_history, chat_turn, turn_score):
    turn_information = pd.Series({'chat_history': chat_history, 'chat_turn': chat_turn, 'turn_score': turn_score})
    chat_turn_information = pd.concat([row.copy().drop(['chat', 'turn_scores']), turn_information])
    return chat_turn_information

#TODO: calcola punteggio medio prima di giocare (come?) e cambia la soglia
def obtain_binary_dataset(dataset):
    #TODO: reduce this better using the game_specific_turn_reward_values
    # TODO: make this values adaptatble to the mean without training

    wordle_score_threshold = 10
    dataset_wordle = dataset[dataset['game'].isin(['wordle'])]
    dataset_wordle = dataset_wordle[~(dataset_wordle['chat_history'].str.len() == 1)]
    dataset_wordle = dataset_wordle[~(dataset_wordle['turn_score'] == wordle_score_threshold)]
    dataset_wordle['label'] = dataset_wordle['turn_score'].apply(
        lambda x: True if x > wordle_score_threshold else False
    )

    wordle_clue_critic_score_threshold = 15
    dataset_wordle_clue = dataset[dataset['game'].isin(['wordle_withclue'])]
    dataset_wordle_clue = dataset_wordle_clue[~(dataset_wordle_clue['chat_history'].str.len() == 1)]
    dataset_wordle_clue = dataset_wordle_clue[~(dataset_wordle_clue['turn_score'] == wordle_clue_critic_score_threshold)]
    dataset_wordle_clue['label'] = dataset_wordle_clue['turn_score'].apply(
        lambda x: True if x > wordle_clue_critic_score_threshold else False
    )

    dataset_wordle_withcritic = dataset[dataset['game'] == 'wordle_withcritic']
    dataset_wordle_withcritic = dataset_wordle_withcritic[~(dataset_wordle_withcritic['chat_history'].str.len() == 1) & (dataset_wordle_withcritic['player'] == 'player 1')]
    dataset_wordle_withcritic['label'] = dataset_wordle_withcritic['turn_score'].apply(
        lambda x: True if x > wordle_clue_critic_score_threshold else False
    )

    #TODO: cambia questo
    dataset_other_games = dataset[~dataset['game'].isin(['wordle', 'wordle_withclue', 'wordle_withcritic'])]
    dataset_taboo_referencegame = dataset_other_games[dataset_other_games['game'].isin(['taboo', 'referencegame'])]
    dataset_taboo_referencegame['label'] = dataset_taboo_referencegame['turn_score'].apply(
        lambda x: True if x > 0 else False
    )

    imagegame_first_turn_threshold = 30
    imagegame_difference_threshold = -5
    dataset_imagegame = dataset_other_games[dataset_other_games['game'].isin(['imagegame'])]
    dataset_imagegame_first_turns = dataset_imagegame[dataset_imagegame['chat_history'].str.len() == 1]
    dataset_imagegame_first_turns['label'] = dataset_imagegame_first_turns['turn_score'].apply(
        lambda x: True if x > imagegame_first_turn_threshold else False
    )

    dataset_imagegame_no_first_turns = dataset_imagegame[~(dataset_imagegame['chat_history'].str.len() == 1)]
    dataset_imagegame_no_first_turns['label'] = dataset_imagegame_no_first_turns['turn_score'].apply(
        lambda x: True if x > imagegame_difference_threshold else False
    )
    dataset_imagegame = pd.concat([dataset_imagegame_first_turns, dataset_imagegame_no_first_turns])


    #TODO: integrate the slot filling part (so fare it only considers points for the ToM part of this game (probe questions))
    privateshared_low_threshold, privateshared_high_threshold, mid_threshold = 0.0, 1.0, 0.5
    dataset_privateshared = dataset_other_games[dataset_other_games['game'].isin(['privateshared'])]
    dataset_privateshared = dataset_privateshared[dataset_privateshared['turn_score'].isin([privateshared_low_threshold, privateshared_high_threshold])]
    dataset_privateshared['label'] = dataset_privateshared['turn_score'].apply(
        lambda x: True if x > mid_threshold else False
    )
    #these lines prevent a huge number of positive samples that create problems for balancing positive and negatives in all games
    dataset_privateshared_positive = dataset_privateshared[dataset_privateshared['label'] == True]
    dataset_privateshared_negative = dataset_privateshared[dataset_privateshared['label'] == False]
    min_rows = min(len(dataset_privateshared_positive), len(dataset_privateshared_negative))
    dataset_privateshared_positive = dataset_privateshared_positive.sample(n=min_rows, random_state=42)
    dataset_privateshared_negative = dataset_privateshared_negative.sample(n=min_rows, random_state=42)
    dataset_privateshared = pd.concat([dataset_privateshared_positive, dataset_privateshared_negative])

    new_dataset = pd.concat([dataset_wordle, dataset_wordle_clue, dataset_wordle_withcritic, dataset_taboo_referencegame, dataset_imagegame, dataset_privateshared], axis=0)

    #TODO: maybe take it out
    #TODO: apply chat template to the prompt (?) or not (?)
    df_binary_dataset = new_dataset.rename(columns={
        'chat_history': 'prompt',
        'chat_turn': 'completion'
    })

    df_binary_dataset['target'] = df_binary_dataset['target'].apply(lambda x: str(x))   #this line solve an error about stored informations

    return df_binary_dataset

def find_positive_and_negative_turns(model_condition, clembench_version):
    turn_format_container = []
    #TODO: uniforma games e row.games
    for game in games:
        # TODO: delete this coondition when referencegame in turn_scores file as well (and not repeat the line df = pd.read_json(file_path, lines=True))
        if game != 'referencegame':
            df = collect_old_and_new_files(game, clembench_version)
        else:
            file_path = f'data/processed/turn_scores/referencegame/{game}_new_processed_10.jsonl'
            df = pd.read_json(file_path, lines=True)

        #TODO: questo si può fare meglio facendo come in dpo_dataset_creator.py (if args.aborted ... dentro il for loop dei games)
        df = df[~((df['Aborted'] == 1) & (df['game'].str.contains('wordle')))]
        df = df[~df['chat'].astype(str).str.contains("Your guess is not a valid word for this game.")]  #this solve an error with few instances not detected as with errors in new benchmark versions of clembench

        if model_condition:
            if model_condition == 'best_models': df = df[df['model'].isin(top_10_models)]
            elif model_condition == 'same_family_model': df = df[df['model'].str.contains('llama', case=False, na=False)]

        for index, row in df.iterrows():
            chat_turns = row['chat']

            scores = row['turn_scores']
            #TODO: questo vale solo per same family and best models, altrimenti togli if (else) e if e metti direttamente il loop
            #TODO: si puo fare con pandas dataframe (solo gli if else, il for tienilo cosi)
            #if row.Lose == 1:    #TODO: rendi un argomento nella funzione
                #if 'llama' in row.model.lower(): #if row.model in top_10_models:      #if 'llama' in row.model.lower():
            #TODO: forse questo if può essere messo nell'else sotto (dopo il for loop e prima della chat_history, con delle modifiche)
            if game == 'wordle_withcritic' and row.player == 'player 1':
                for i in range(len(chat_turns)):
                    if (i + 1) % 4 == 0:
                        chat_history = chat_turns[:i]
                        chat_turn = [chat_turns[i]]
                        turn_score = scores[str((i+1)//4)]['strategy score']
                        #TODO: reduce this and the laters
                        chat_turn_information = create_turn_information(row, chat_history, chat_turn, turn_score)
                        turn_format_container.append(chat_turn_information)
                    else: continue
            elif game == 'privateshared':
                for i in range(len(scores.values())):
                    #chat_history = chat_turns[:i*(10+2)+1] #privateshared game has this structure: instruction + 10ToM QA + 2QA and we are interested in the 10ToM QA
                    probe_questions = chat_turns[i*(10+2)+1:i*(10+2)+11]
                    turn_score = scores[str(i)]['Accuracy']
                    for idx_questions in range(1, len(probe_questions) // 2 + 1):
                        chat_history = chat_turns[:i * (10 + 2) + idx_questions*2]
                        chat_turn = [chat_turns[i * (10 + 2) + idx_questions*2]]
                        chat_turn_information = create_turn_information(row, chat_history, chat_turn, turn_score)
                        turn_format_container.append(chat_turn_information)
            else:
                for i in range(1, len(chat_turns) // 2 + 1):
                    chat_history = chat_turns[:i + i - 1]
                    chat_turn = [chat_turns[i + i - 1]]
                    # TODO: 4 linee, non controllare cosi ma ad ogni gioco una volta sola
                    if 'wordle' in game:
                        if game == 'wordle_withcritic' and row.player == 'player 2':
                            #TODO: per ora saltati questi casi e non considerati, vedi come integrare
                            continue
                        else:
                            turn_score = scores[str(i)]['strategy score']
                    elif row.game == 'imagegame' and bool(scores):
                        if i != len(chat_turns) // 2:
                            if i == 1:
                                turn_score=scores[str(i-1)]['F1']
                            else:
                                turn_score=scores[str(i - 1)]['F1'] - scores[str(i - 2)]['F1']
                    elif row.game == 'referencegame' and bool(scores):
                        turn_score = scores[str(i-1)]['Success']
                    elif bool(scores): # taboo
                        turn_score = scores[str(i-1)]['Accuracy']
                        assert row.game == 'taboo'
                        if row.player =='player 1':
                            assert len(chat_turn) == 1
                        else:
                            if row.benchmark_version in ['v0.9', 'v1.0']:
                                chat_history = chat_turns[:i + i]
                                chat_turn = [chat_turns[i + i]]
                        chat_turn[0]['role'] = 'assistant'
                    else: continue

                    chat_turn_information = create_turn_information(row, chat_history, chat_turn, turn_score)
                    turn_format_container.append(chat_turn_information)
    return turn_format_container


#TODO: omologa con la precedente
def find_aborted_turns_wordle(model_condition):
    aborted_format_container = []
    for file in file_paths_old_withaborted:
        df = pd.read_json(file, lines=True)

        if model_condition:
            if model_condition == 'best_models': df = df[df['model'].isin(top_10_models)]
            elif model_condition == 'same_family_model': df = df[df['model'].str.contains('llama', case=False, na=False)]

        df_aborted = df[df.Aborted==1]

        for index, row in df_aborted.iterrows():
            chat_turns = row['chat']
            for i, chat_turn in enumerate(chat_turns):
                if chat_turn['has_error'] == True:
                    chat_history = list(map(lambda d: {k: v for k, v in d.items() if k != 'has_error'}, chat_turns[:i]))
                    chat_turn = [{'content':chat_turn['content'], 'role':chat_turn['role']}]
                    #TODO: controlla che funzioni queste due righe cancellate
                    #turn_information = pd.Series({'chat_history':chat_history, 'chat_turn':chat_turn, 'turn_score':-10000}) #-10000 'fake' value for strategic score of error turns in aborted games
                    #chat_turn_information = pd.concat([row.copy().drop(['chat', 'turn_scores']), turn_information])
                    chat_turn_information = create_turn_information(row, chat_history, chat_turn, -10000) #-10000 'fake' value for strategic score of error turns in aborted games
                    aborted_format_container.append(chat_turn_information)
                    #TODO: if you remove the break point here, you can get ALL the negatives and not just THE FIRST
                    #break
    return aborted_format_container


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create DPO datasets')
    parser.add_argument('--model_condition', default=False, choices=[False, 'best_models', 'same_family_model'], help='restriction to the negative samples to be from best models or from the family of the model to train (llama)')
    parser.add_argument('--aborted_interactions', default=True, choices=[True, False], help='integrating aborted interactions as negative samples')
    parser.add_argument('--clembench_version', default='old', choices=['old', 'old_and_new'], help='clembench versions employed')
    #TODO: take this out in common with DPO_training.py and KTO_training.py
    parser.add_argument('--hf_login', default="", help='hf login token')
    parser.add_argument('--hf_repo', default='clembench-playpen', help='huggingface repository to store the created datasets')
    args = parser.parse_args()

    login(f"{args.hf_login}")

    turn_format_container = find_positive_and_negative_turns(model_condition=args.model_condition, clembench_version=args.clembench_version)
    aborted_format_container = find_aborted_turns_wordle(model_condition=args.model_condition)
    format_container_turn_aborted = turn_format_container + aborted_format_container
    df_turn = pd.DataFrame(format_container_turn_aborted)
    df_binary_dataset = obtain_binary_dataset(df_turn)
    hf_binary_dataset = Dataset.from_pandas(df_binary_dataset)
    hf_binary_dataset.push_to_hub(f"clembench-playpen/KTO_{'Aborted' if args.aborted_interactions else ''}{'_'+args.model_condition if args.model_condition else ''}_{args.clembench_version}")
