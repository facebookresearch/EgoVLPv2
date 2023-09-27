## 📝 EgoTaskQA Data Preparation

We follow the [official EgoTaskQA repository](https://github.com/Buzz-Beater/EgoTaskQA/tree/main/baselines) for data download and preprocessing. For quickstart, the preprocessed videos and annotations can be downloaded as:

```bash
wget https://www.cis.jhu.edu/~shraman/EgoVLPv2/datasets/EgoTaskQA/qa_videos.tgz
wget https://www.cis.jhu.edu/~shraman/EgoVLPv2/datasets/EgoTaskQA/Data.tgz
tar -xvzf qa_videos.tgz && rm qa_videos.tgz
tar -xvzf Data.tgz && rm Data.tgz
wget https://www.cis.jhu.edu/~shraman/EgoVLPv2/datasets/EgoTaskQA/reasoning_unique_cat.pth
```
## 📊 Fine-tuned Checkpoints and Results

| Method | Split-type  | #Epochs | Peak lr | Checkpoint | Open | Binary | All |
| :----: | :-----: | :--------: | :---------: | :------: | :-------: | :-------: | :-------: | 
| EgoVLPv2 | Direct | 36 | 2e-4 | [EgoVLPv2_EgoTaskQA_Direct](https://www.cis.jhu.edu/~shraman/EgoVLPv2/ckpts/EgoTaskQA_Finetuned/EgoTaskQA_finetune_direct.tar) | 35.56 | 75.60 | 46.26 | 
| EgoVLPv2 | Indirect | 36 | 1e-4 | [EgoVLPv2_EgoTaskQA_Indirect](https://www.cis.jhu.edu/~shraman/EgoVLPv2/ckpts/EgoTaskQA_Finetuned/EgoTaskQA_finetune_indirect.tar) | 29.14 | 59.68 | 42.28 | 


## 🎯 Fine-tuning on EgoTaskQA
Use the pre-trained [EgoVLPv2 model](https://www.cis.jhu.edu/~shraman/EgoVLPv2/ckpts/Pre-trained/EgoVLPv2.pth) and fine-tune separately for the direct and indirect splits of EgoTaskQA. The split type is controlled by the `--dataset_split_type` argument. We use a peak learning rate of 2e-4 for the direct split, and 1e-4 for the direct split with linear warmup and cosine decay. The 36 epoch fine-tuning takes ~20 hours using 8 V100/A5000 cards. 

```bash
python main_end2end.py --dataset_split_type direct --model_name EgoVLPv2.pth --per_gpu_batch_size 8 --num_frames_per_video 16 --frame_resolution 224 --lr 2e-4
python main_end2end.py --dataset_split_type indirect --model_name EgoVLPv2.pth --per_gpu_batch_size 8 --num_frames_per_video 16 --frame_resolution 224 --lr 1e-4
```

Due to an unpresedented situation, if the fine-tuning stops before completion, the process can be resumed from last saved checkpoint. Mention the path of last saved checkpoint using `--resume_finetune_model_path` argument. 

```bash
python main_end2end.py --dataset_split_type direct --model_name EgoVLPv2.pth --resume_finetune_model_path <last_saved_ckpt> --per_gpu_batch_size 8 --num_frames_per_video 16 --frame_resolution 224 --lr 2e-4
```

## ⚓️ Evaluate Fine-tuned Checkpoint (test-only)
To evaluate the fine-tuned checkpoint, mention the model path using `--test_only_model_path` argument.

```bash
python main_end2end.py --dataset_split_type direct --test_only_model_path EgoTaskQA_finetune_direct.tar --per_gpu_batch_size 8 --num_frames_per_video 16 --frame_resolution 224
python main_end2end.py --dataset_split_type indirect --test_only_model_path EgoTaskQA_finetune_indirect.tar --per_gpu_batch_size 8 --num_frames_per_video 16 --frame_resolution 224
```

## 🙏 Acknowledgement
We thank the [EgoTaskQA](https://arxiv.org/abs/2210.03929) authors for releasing the dataset and baselines.
