import os
import json
import xmltodict
import uuid

label_dict = {
    '매수': 'BUY',
    '중립': 'HOLD',
    '보유': 'HOLD',
    '' : 'NOT RATED'
}

src_html_dir = 'datasets/src_html'
src_gen_dir = 'datasets/src_gen'

def transform_html(src_file_name):
    with open(os.path.join(src_html_dir, src_file_name), 'r') as f:
        aa = f.read()
    bb = aa.replace("<br>"," ").replace("ㅁ","")
    cc = xmltodict.parse(bb)
    dd = json.loads(json.dumps(cc))

    result = list()
    for tr in dd['tbody']['tr']:
        try:

            # Use the random_id variable as needed
            random_id = str(uuid.uuid4())[:4]
            std_dt = tr['td'][0]['nobr']
            rep_secn = tr['td'][1]['nobr'][-7:-1]
            id = f"{std_dt}_{rep_secn}_{random_id}"
            summary = tr['td'][2]['nobr']
            label_ = tr['td'][3]['nobr']
            label = label_dict.get(label_, label_.upper())
            item = {
                "id": id,
                "document": summary,
                "label": label
            }
            result.append(item)
        except Exception as e:
            pass

    with open(f'{src_gen_dir}/{src_file_name.split(".")[0]}.out', 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    
def main():
    os.makedirs(src_html_dir, exist_ok=True)
    os.makedirs(src_gen_dir, exist_ok=True)
    
    for file_name in os.listdir(src_html_dir):
        transform_html(file_name)

if __name__ == "__main__":
    main()