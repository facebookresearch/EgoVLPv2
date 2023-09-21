## 📝 EgoMQ Data Preparation

The EgoMQ metadata can be downloaded from the [Ego4D official webpage](https://ego4d-data.org/). Follow the annotation conversion step [here](https://github.com/EGO4D/episodic-memory/tree/main/MQ#annotation-conversion). Keep the metadata in `jsons/` folder.
For quickstart, the matadata can be easily downloaded as follows:

```bash
wget https://www.cis.jhu.edu/~shraman/EgoVLPv2/datasets/EgoMQ/jsons.tgz
tar -xvzf jsons.tgz && rm jsons.tgz
```

## 📊 Results

| Method | mAP (%) @ IoU=0.1 | mAP (%) @ IoU=0.3 | mAP (%) @ IoU=0.5 | mAP (avg) | 
| :----: | :-----: | :--------: | :---------: | :------: | 
| EgoVLPv2 + VSLNet | 17.58 | 11.92 | 6.90 | 12.23 |
 

## ⚙️ Pre-extracted EgoMQ Features

Our pre-extracted video features can be downloaded as: 
```bash
mkdir saved_features 
cd saved_features
wget https://www.cis.jhu.edu/~shraman/EgoVLPv2/pre-extracted_features/EgoMQ/EgoVLPv2.tgz
tar -xvzf EgoVLPv2.tgz && rm EgoVLPv2.tgz
```

## 🎯 Fine-tuning on EgoMQ
This script uses PyTorch’s DataParallel (DP) implementation. For feature extraction, please follow these [steps](https://github.com/ShramanPramanick/EgoVLPv2/tree/main/EgoVLPv2#%EF%B8%8F-feature-extraction-on-egomq). To run head-tuning, modify the `Features` variable in `scripts/train_infer_eval_ego_nce.sh` with proper path of extracted features. 

```bash
# We perform a grid-search for four different hyper-parameters: batch_size, learning_rate, step_size, and step_gamma.
bash scripts/train_infer_eval_ego_nce.sh
```
The evaluation results will be saved in `/outputs/` directory. 
```bash
# Find the best results
python find_best_parameters.py --root_folder ./outputs/
```

## 🙏 Acknowledgement
We use [VSGN](https://arxiv.org/abs/2011.14598) as task-specific head for EgoMQ. 
