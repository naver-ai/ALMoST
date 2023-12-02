## Evaluation

We provide evaluation code for the three benchmarks we reported.

- hhh (Static HHH eval)
- truthful (TruthfulQA-MC1)
- vicuna (Vicuna Eval w. GPT-4 Eval)

Please use the same environmental setup of training!

* `source almost_train/bin/activate`


### 1. Static HHH Eval

```
python run_evaluation.py \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --benchmark_name hhh
```

### 2. TruthfulQA-MA1

```
python run_evaluation.py \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --benchmark_name truthful
```

### 3. Vicuna Eval (GPT-4 Eval)

We used legacy version of Vicuna Evaluation provided in [FastChat (v.0.2.1)](https://github.com/lm-sys/FastChat/tree/v0.2.1/fastchat/eval). <br>
You can compare the models with the latest version. <br>

If you want to reproduce our results, please follow the below descriptions. <br>
Please note that it is not fully reproducible because of the stochasticity of GPT-4 response.

```
python run_evaluation.py \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --benchmark_name vicuna
  --baseline_model_name $BASELINE_MODEL_NAME
```
