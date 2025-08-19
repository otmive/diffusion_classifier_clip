# **Evaluating Compositional Generalisation in VLMs and Diffusion Models**

This repo is for the paper Evaluating Compositional Generalisation in VLMs and Diffusion Models.

This work uses the Diffusion-Classifier model proposed in [Your Diffusion Model is Secretly a Zero-Shot Classifier](https://github.com/diffusion-classifier/diffusion-classifier)

Create environment
```
conda env create -f enrironment.yml
```
Install clip
```
pip install git+https://github.com/openai/CLIP.git
```

## Load dataset
You can download the dataset from https://drive.google.com/file/d/14U4azHV6FHI8yeALWfgKvnH40OpDvbz1/view?usp=drive_link

## Fine-tune CLIP
```
python clip_finetune.py --data_path <path_to_training_images> --dataset <single|two_object|relational> --seed <seed_number> --save_path <saved_weights.pt>
```
For example:
`python clip_finetune.py --data_path "cobi2_datasets/single_object/train" --dataset "single" --seed 1 --save_path "single_ft.pt"`

## Fine-tune Diffusion-Classifier
```
python train_dreambooth.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
  --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
  --output_dir=<save_folder_name> \
  --revision="fp16" \
  --seed=1 \
  --resolution=512 \
  --train_batch_size=1 \
  --train_text_encoder \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="polynomial" \
  --lr_warmup_steps=0 \
  --num_class_images=30 \
  --sample_batch_size=4 \
  --max_train_steps=10 \
  --save_interval=10 \
  --data_type=<single|two_object|relational> \
  --folder_path=<path_to_training_images> \
```


## Run models on data
### CLIP
You can use for example the following command to run frozen CLIP on the ID val single data split
```python clip_predict.py --image_folder 'cobi2_datasets/single_object/ID_val/' --output_file 'results/single/clip/clip_single_idval_s1.csv' --dataset single
```

You can use for example the following command to run the fine-tuned single model on the ID val single data split
```python clip_predict.py --image_folder 'cobi2_datasets/single_object/ID_val/' --output_file 'results/single/clip/clip_single_idval_s1' --dataset single --model_path models/clip/single_object/seed_1_single.pt
```
For two object and relational the prompt path must be specified:
```
python clip_predict.py --image_folder 'cobi2_datasets/two_object/ood_val/' --output_file 'clip_two_oodval_frz' --dataset two_object --prompt_path "cobi2_datasets/two_object/two_object_prompts/two_obj_val"
```

### Diffusion Classifier
You can use the following command to run Diffusion Classfiier on the ID val single data split:
```
python diffusion-classifier/eval_prob_adaptive.py \
--dataset clevr \
--split test \
--n_trials 1 \
--to_keep 5 1 \
--n_samples 75 200 \
--loss l1 \
--prompt_path cobi2_datasets/single_object/single_prompts.csv \
--seed 1 \
--dataset_path 'cobi2_datasets/single_object/ID_test/' \
--output_file 'single_object_id_val.csv' \
--data_split 'id_val' \
--data_type 'single'
#--model_path 'models/single_object/seed_1.pt'
```
The `model_path` flag can be used to load fine-tuned models. 

Two object and relational predictions can be run using the corresponding folder of prompt files, for example:
```
python diffusion-classifier/eval_prob_adaptive.py \
--dataset clevr \
--split test \
--n_trials 1 \
--to_keep 5 1 \
--n_samples 75 200 \
--loss l1 \
--prompt_path cobi2_datasets/two_object/two_object_prompts/two_obj_val \
--seed 1 \
--dataset_path 'cobi2_datasets/two_object/ood_val/' \
--output_file 'two_object_ood_val' \
--data_split 'ood_val' \
--data_type 'two_object'
```
### Viewing Results 

The results are saved as csv files of predictions. To view the accuracy using the outputs the `print_acc.py' file can be used. 
```
python print_acc.py --output_file "results/single/clip/clip_single_idval_s1.csv" --dataset "single"
```
For two object and relational the dataset split must be specified: idval, idtest, val, test, idval_gen, idtest_gen, val_gen or test_gen.
```
python print_acc.py --output_file "two_object_ood_val" --dataset "two_object" --datatype "val"
```