## 📝 EgoNLQ Data Preparation

The EgoNLQ metadata can be downloaded from the [Ego4D official webpage](https://ego4d-data.org/). Follow the data preparation steps [here](https://github.com/EGO4D/episodic-memory/tree/main/NLQ/VSLNet#preparation). Keep the metadata in `jsons/` folder.
For quickstart, the matadata can be easily downloaded as follows:

```bash
wget https://www.cis.jhu.edu/~shraman/EgoVLPv2/datasets/EgoNLQ/jsons.tgz
tar -xvzf jsons.tgz && rm jsons.tgz
```

## 📊 Results

| Method | Max_Pos_Len | Batch_Size | Init_lr | mIOU@0.3 (R@1) | mIOU@0.3 (R@5) | mIOU@0.5 (R@1) | mIOU@0.5 (R@5) | 
| :----: | :-----: | :--------: | :---------: | :------: | :-------: | :-------: | :-------: | 
| EgoVLPv2 + VSLNet | 256 | 32 | 0.001 | 12.95 | 23.80 | 7.91 | 16.11 |

Our EgoNLQ head-tuning log can be found [here](https://www.cis.jhu.edu/~shraman/EgoVLPv2/logs/EgoNLQ_Head-tune_log_256_32_0.001.txt). These results are generated using 4-frame head-tuning. We also perform 16-frame counterpart which is reasonably more time consuming without a significant performance improvement. 

## ⚙️ Pre-extracted EgoNLQ Features

Our pre-extracted video features can be downloaded as: 
```bash
mkdir saved_features && mkdir saved_nlq_results && mkdir saved_video_frames
cd saved_features
wget https://www.cis.jhu.edu/~shraman/EgoVLPv2/pre-extracted_features/EgoNLQ/EgoVLPv2.tgz
tar -xvzf EgoVLPv2.tgz && rm EgoVLPv2.tgz
```


## 🎯 Fine-tuning on EgoNLQ
This script uses PyTorch’s DataParallel (DP) implementation. If the pre-loaded frames and pre-extracted features are present, this script will skip those steps and start head-tuning. Else, the script will extract features first, and then will perform head-tuning. 

```bash
python main.py --max_pos_len 256 --batch_size 32 --init_lr 0.001 --epochs 50 --cuda_base cuda:0 --device_ids 0,2,3,4,5,6,7 --model_name EgoVLPv2.pth --predictor EgoVLP --video_feature_dim 768 --eval_gt_json ./jsons/nlq_val.json
```
We also provide a smaller head in `model/VSLNet_small.py`, which can be used to reproduce the (EgoVLPv2 + QGH + Span) row in Table F.1. of our [paper](https://arxiv.org/pdf/2307.05463.pdf).

## 🙏 Acknowledgement
We use [VSLNet](https://arxiv.org/pdf/2004.13931.pdf) as task-specific head for EgoNLQ. 
