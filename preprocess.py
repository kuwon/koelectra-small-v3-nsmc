import os
import json
import sys
import argparse
from datasets import Dataset, load_from_disk

src_gen_dir = 'datasets/src_gen'
dataset_path = "YOUR_DATASET_PATH"

opinion_list = list()
for one_html in os.listdir(src_gen_dir):
    with open(os.path.join(src_gen_dir, one_html), 'r') as f:
        aa = json.load(f)
    for one_opinion in aa:
        opinion_list.append(one_opinion)



def gen_invst_opinion_grd_content():
    for one_opinion in opinion_list:
        yield one_opinion

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default='invst_opinion', type=str, help="Dataset Name")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parser_args()
    ds_name = args.name
    os.environ['GPU_NUM_DIVICES'] = '1'

    Dataset.from_generator(gen_invst_opinion_grd_content).save_to_disk(f"datasets/{ds_name}")
    samples = load_from_disk(f"datasets/{ds_name}").train_test_split(test_size=0.2)
    print(samples)
    samples.push_to_hub("solikang/invest_opinion_1w_sample", private=True)
    json_file_path = f'{dataset_path}/{ds_name}/dataset_info.json'

    # Read the JSON file
    with open(json_file_path, 'r') as json_file:
        dataset_dict = json.load(json_file) 

    cols = list(dataset_dict['features'].keys()) 
    print(cols)
    print(samples['train'].features)
    print(samples['test'][0])