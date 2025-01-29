import argparse
from cmath import inf
from datasets import Dataset
import pandas as pd
from collections import defaultdict
from huggingface_hub import login

file_paths_old = ['data/processed/wordle_old_processed.jsonl',
              'data/processed/wordle_withclue_old_processed.jsonl',
              'data/processed/wordle_withcritic_old_processed.jsonl',
              'data/processed/taboo_old_processed.jsonl',
              'data/processed/referencegame_old_processed.jsonl',
              'data/processed/privateshared_old_processed.jsonl',
              'data/processed/imagegame_old_processed.jsonl',]

top_10_models = [
    'gpt-4-0613-t0.0--gpt-4-0613-t0.0',
    'claude-v1.3-t0.0--claude-v1.3-t0.0',
    'gpt-4-1106-preview-t0.0--gpt-4-1106-preview-t0.0',
    'gpt-4-t0.0--gpt-4-t0.0',
    'gpt-4-0314-t0.0--gpt-4-0314-t0.0',
    'claude-2.1-t0.0--claude-2.1-t0.0',
    'gpt-4-t0.0--gpt-3.5-turbo-t0.0',
    'claude-2-t0.0--claude-2-t0.0',
    'gpt-3.5-turbo-1106-t0.0--gpt-3.5-turbo-1106-t0.0',
    'gpt-3.5-turbo-0613-t0.0--gpt-3.5-turbo-0613-t0.0',
]

def match_successful_unsuccessful(df_successful_games, df_unsuccessful_games, match_count, model_condition):
    df_successful = df_successful_games.copy()
    df_unsuccessful = df_unsuccessful_games.copy()
    df_successful['first_message'] = df_successful['chat'].apply(lambda x: x[0]['content'])
    df_unsuccessful['first_message'] = df_unsuccessful['chat'].apply(lambda x: x[0]['content'])
    match_cols = ['game', 'game_id', 'benchmark_version', 'experiment', 'episode', 'first_message', 'target']
    match_cols += ['player'] if 'player' in df_successful_games.columns else []
    df_successful['merge_key'] = df_successful[match_cols].apply(lambda x: '_'.join(str(val) for val in x), axis=1)
    df_unsuccessful['merge_key'] = df_unsuccessful[match_cols].apply(lambda x: '_'.join(str(val) for val in x), axis=1)
    result_dict = defaultdict(list)
    for _, succ_row in df_successful.iterrows():
        matching_unsucc = df_unsuccessful[df_unsuccessful['merge_key'] == succ_row['merge_key']]
        if match_count != float(inf): matching_unsucc = matching_unsucc.head(int(match_count))
        if model_condition:
            if model_condition == 'best_models': matching_unsucc = matching_unsucc[matching_unsucc['model'].isin(top_10_models)]
            elif model_condition == 'same_family_model': matching_unsucc = matching_unsucc[matching_unsucc['model'].str.contains('llama', case=False, na=False)]
        for _, unsucc_row in matching_unsucc.iterrows():
            result_dict['game'].append(succ_row['game'])
            result_dict['game_id'].append(succ_row['game_id'])
            result_dict['benchmark_version'].append(succ_row['benchmark_version'])
            result_dict['experiment'].append(succ_row['experiment'])
            result_dict['episode'].append(succ_row['episode'])
            result_dict['model_successful'].append(succ_row['model'])
            result_dict['model_unsuccessful'].append(unsucc_row['model'])
            result_dict['prompt'].append(succ_row['first_message'])
            result_dict['chosen'].append(succ_row['chat'])
            result_dict['rejected'].append(unsucc_row['chat'])
    return dict(result_dict)

def matches_all_files(file_paths, match_count, model_condition):
    success_and_lose_all_games = defaultdict(list)
    for o in file_paths:
        df = pd.read_json(o, lines=True)
        df_success = df[df.Success == 1]
        df_lose = df[df.Lose == 1]
        game_matches = match_successful_unsuccessful(df_success, df_lose, match_count, model_condition)
        for key in game_matches:
            success_and_lose_all_games[key].extend(game_matches[key])
    return dict(success_and_lose_all_games)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create DPO datasets')
    parser.add_argument('--neg', default=float(inf), type=float, help='number of negative samples per every positive one')
    parser.add_argument('--model_condition', default=False, choices=[False, 'best_models', 'same_family_model'], help='restriction to the negative samples to be from best models or from the family of the model to train (llama)')
    #TODO: take this out in common with DPO_training.py and KTO_training.py
    parser.add_argument('--hf_login', help='hf login token')
    parser.add_argument('--hf_repo', default='clembench-playpen', help='huggingface repository to store the created datasets')
    args = parser.parse_args()

    login(f"{args.hf_login}")
    success_and_lose_all_games = matches_all_files(file_paths_old, match_count=args.neg, model_condition=args.model_condition)
    hf_dataset = Dataset.from_dict(success_and_lose_all_games)
    hf_dataset.push_to_hub(f"{args.hf_repo}/DPO_{args.neg if args.neg != float(inf) else 'all'}neg{'_'+args.model_condition if args.model_condition else None}")