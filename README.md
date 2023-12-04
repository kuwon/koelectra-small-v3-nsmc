# koelectra-small-v3-nsmc
분석리포트 투자의견 분석을 위한 Fine-Tuned LLM model 생성 PoC


## 1. Reformat source data
- html(from src_html) to json(to src_gen) with xmltodict
``` bash
python3 ./parse_tbody_to.py
```

## 2. Generate Datasets
- reformat to hugging-face datasets
``` bash
python3 ./preprocess.py
```

## 3. Train models
- fine-tuning pre-trained models with dataset
``` bash
python3 ./train.py
```

## 4. Test models
- inference model
``` bash
python3 ./test.py
```