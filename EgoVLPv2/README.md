## 📝 Data Preparation

### EgoClip & EgoMCQ

We follow the [EgoVLP](https://github.com/showlab/EgoVLP/tree/3919d73149c0547b8383d22d8b370c71a1905a49#-preparation) for downloading and preprocessing Ego4D. Go through the following steps for preparing pre-training and validation data:
1. Download Ego4D videos and metadata
Follow the guideline here, download the following to {PATH_TO_EGO4D}. Create `data_root` add a soft link by `ln -s {PATH_TO_EGO4D} data_root/ego4d`.
2. Preprocess Ego4D videos
For effectively pretraining, we follow EgoVLP and compress videos in the following way:
    - Resize the source videos with a short size equal to 256 by script `utils/video_resize.py`.
    - Chunk the resized videos to multiple segments (up to 600 sec) by script `utils/video_chunk.py`.
3. Prepare EgoClip metadata
```bash
gdown --fuzzy https://drive.google.com/file/d/1-aaDu_Gi-Y2sQI_2rsI2D1zvQBJnHpXl/view
```
4. Prepare EgoMCQ metadata
```bash
gdown --fuzzy https://drive.google.com/file/d/1-5iRYf4BCHmj4MYQYFRMY4bhsWJUN3rW/view
```

Ego4D videos should be structured as:

```bash
$data_root/
    Ego4D_Chunked/
         fffbaeef-577f-45f0-baa9-f10cabf62dfb/
              0.mp4
              1.mp4
         fff5b8bd-1fc2-457b-9760-7691a7c1d095/
              0.mp4
              1.mp4
              2.mp4
              ...
         ...       
```

### EPIC-Kitchens-100 (EK-100)
We follow [LAVILA](https://github.com/facebookresearch/LaViLa/blob/main/datasets/README.md#epic-kitchens-100-ek-100) for preparing EK-100 annotations and resized videos. 

1. Download annotations

```bash
git clone https://github.com/epic-kitchens/epic-kitchens-100-annotations
```

2. Download videos.

    a. For raw videos, please download them from [https://epic-kitchens.github.io/](https://epic-kitchens.github.io/). Academic Torrents works well for downloading raw videos. You can find [EPIC-KITCHENS-55](https://academictorrents.com/details/d08f4591d1865bbe3436d1eb25ed55aae8b8f043) here and [EPIC-KITCHENS-100](https://academictorrents.com/details/cc2d9afabcbbe33686d2ecd9844b534e3a899f4b) here.

    b. (Recommended) The raw videos are huge (~1.1 TB). As an alternative, please check out a [resized version](https://utexas.box.com/s/l7ij81ie5q07p9fdg0vtejihq61liln9).

3. (For EK-100 MIR)

    a. Generate the relevancy matrix of train/val splits using [the official code](https://github.com/mwray/Joint-Part-of-Speech-Embeddings).

    b. (Recommended) The generated result has some randomness. Therefore, LAVILA provide the [replica of train split](https://dl.fbaipublicfiles.com/lavila/metadata/EK100/caption_relevancy_EPIC_100_retrieval_train.pkl) and [val split](https://dl.fbaipublicfiles.com/lavila/metadata/EK100/caption_relevancy_EPIC_100_retrieval_test.pkl). Please put them to the folder `$data_root/EK100/epic-kitchens-100-annotations/retrieval_annotations/relevancy/`.


The folder should look like this:
```bash
$data_root/
    EK100/
        epic-kitchens-100-annotations/
            EPIC_100_train.csv
            EPIC_100_validation.csv
            ...
            retrieval_annotations/relevancy/  # this appears if you do 3.
                caption_relevancy_EPIC_100_retrieval_train.pkl
                caption_relevancy_EPIC_100_retrieval_test.pkl
        video_ht256px/
            P01/
                P01_01.MP4
                P01_02.MP4
                ...
                P01_19.MP4
            P02/
                P02_01.MP4
                P02_02.MP4
                ...
                P02_15.MP4
            ...
```

### Charades-Ego

1. Download annotations and 480p videos from [https://prior.allenai.org/projects/charades-ego](https://prior.allenai.org/projects/charades-ego).

2. Generate additional filtered metadata files `metadata_train.csv`, `metadata_val.csv` and `metadata_test.csv` using `utils/charades_meta.py`. 

**[QuickStart]** For quickstart, we provide the completely processed CharadesEgo dataset. Download it as follows:

```bash
# cd to your data_root
wget http://www.cis.jhu.edu/~shraman/EgoVLPv2/datasets/CharadesEgo.tgz
tar -xvzf CharadesEgo.tgz && rm CharadesEgo.zip
```

The folder should look like this:
```bash
$data_root/
    CharadesEgo/
        CharadesEgo/
            CharadesEgo_v1_train_only1st.csv
            CharadesEgo_v1_test_only1st.csv
            ...
            metadata_train.csv  # this appears if you do 2.
            metadata_val.csv # this appears if you do 2.
            metadata_test.csv # this appears if you do 3.
        CharadesEgo_v1_480/
            005BU.mp4
            005BUEGO.mp4
            ...
```

### EgoMQ (feature extraction)

For EgoMQ feature extraction, you need Ego4D resized but non-chunked videos, which will be the output of the first step of video pre-processing (see first bullet of preprocessing [here](https://github.com/ShramanPramanick/EgoVLPv2/blob/main/EgoVLPv2/README.md#egoclip--egomcq)). The videos will be structured as follows:
```bash
$data_root/
    Ego4D_256/
        fffbaeef-577f-45f0-baa9-f10cabf62dfb.mp4
        fff5b8bd-1fc2-457b-9760-7691a7c1d095.mp4
        ...
```
Update the `data_dir` in `configs/eval/mq.json`.

## 📊 Pre-trained Checkpoints and Results on EgoMCQ

| Method  | Epochs | Projector Dimension | Frames <br /> (Pre-training) | Frames <br /> (Validation) | Checkpoint | EgoMCQ Inter-video Acc. |  EgoMCQ Intra-video Acc. | 
| :-----: | :--------: | :--------: | :----------: | :---------: | :---------: | :------: | :------: |
| EgoVLPv2 | 20 |  768-4096-4096-4096 |  4 | 16 | [EgoVLPv2](http://www.cis.jhu.edu/~shraman/EgoVLPv2/ckpts/Pre-trained/EgoVLPv2.pth) | 91.0 | 60.9 |

- The above checkpoint is used for EgoMCQ, EgoNLQ, EgoMQ, EgoTaksQA, and QFVS. However, for EK-100 and CharadesEgo, we find that pre-trained model with smaller projector (768-256) works better (LAVILA also noticed the same; see Table 11.c. in [LAVILA paper](https://arxiv.org/pdf/2212.04501.pdf). Moreover, following the findings of EgoVLP, we also observe that the zero-shot performance on Charades-Ego significantly degrades after the first pre-training epoch. Hence, please use [EgoVLPv2_smallproj](http://www.cis.jhu.edu/~shraman/EgoVLPv2/ckpts/Pre-trained/EgoVLPv2_smallproj.pth) for zero-shot EK-100, and use [EgoVLPv2_smallproj_epoch1](http://www.cis.jhu.edu/~shraman/EgoVLPv2/ckpts/Pre-trained/EgoVLPv2_smallproj_epoch1.pth) for zero-shot CharadesEgo.
- Our pre-training log can be found [here](http://www.cis.jhu.edu/~shraman/EgoVLPv2/logs/Pre-training_EgoVLPv2_stats.txt). Note that, after every pre-training epoch, the evaluation is done with 4 frames to save time. The best model is received after 10th epoch, which produces 91.0% and 60.9% inter- and intra-video accuracy on EgoMCQ when evaluated using 16 frames.
- Feel free to use our pre-trained checkpoints for other egocentric benchmarks.

## ⚓️ Evaluation on EgoMCQ

Download [EgoVLPv2](http://www.cis.jhu.edu/~shraman/EgoVLPv2/ckpts/Pre-trained/EgoVLPv2.pth) checkpoint, and Update `load_checkpoint` in `configs/eval/egoclip.json` with its path. 

```bash
python multinode_train_egoclip.py --save_dir validation_egoVLPv2 --config ./configs/eval/egomcq.json --print_freq 100
```

## 🏋️‍️ Pretraining
We use PyTorch’s native DistributedDataParallel (DDP), FP16 mixed precision training and gradient checkpointing. We pre-train our model for 20 epochs with a batch size of 256, using AdamW with a peak learning rate of 3e-5 for the backbones and 12e-5 for the cross-modal parameters. We use linear warmup over the first 2 epochs and use linear decay. Pre-training takes ~5 days on 32 A100 GPUs.

Start by modifying `configs/pt/egoclip.json` and `configs/eval/egoclip.json` with proper datapath and checkpoint paths. Next, download TimeSFormer initialization from [here](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth) and modify `model/model.py` with proper initialization path.  

- Pre-train from command line (interactive job, only supports single-node):
```bash
python multinode_train_egoclip.py --save_dir results_egoVLPv2 --config ./configs/pt/egoclip.json --print_freq 100
```

- Pre-train using slurm (supports multi-node): 

```bash
# Configure slurm parameters in run.sh & launch the job
sbatch run.sh
```
## 📊 Fine-tuned Checkpoints and Results on EK-100

| Method | Eval | Checkpoint | mAP | nDCG |
| :----: | :-----: | :--------: | :---------: | :------: | 
| EgoVLPv2 | Zero-Shot | [EgoVLPv2_smallproj](http://www.cis.jhu.edu/~shraman/EgoVLPv2/ckpts/Pre-trained/EgoVLPv2_smallproj.pth) | 26.7 | 29.1 |
| EgoVLPv2 | Fine-tune | [EgoVLPv2_EK-100](http://www.cis.jhu.edu/~shraman/EgoVLPv2/ckpts/EK-100_Finetuned/EK-100_finetune.pth) | 47.3 | 61.9 |


## 🎯 Fine-tuning on EK-100

Download [EgoVLPv2_smallproj](http://www.cis.jhu.edu/~shraman/EgoVLPv2/ckpts/Pre-trained/EgoVLPv2_smallproj.pth) checkpoint, and Update `load_checkpoint` in `configs/ft/epic.json` with its path.

```bash
python multinode_train_epic.py --save_dir finetune_epic --config ./configs/ft/epic.json --print_freq 100
```

For fine-tuning on multi-nodes, prepare a slurm script similar to `run.sh`.

## ⚓️ Zero-shot and Fine-tuned Checkpoint Evaluation on EK-100 (test-only)

Download [EgoVLPv2_smallproj](http://www.cis.jhu.edu/~shraman/EgoVLPv2/ckpts/Pre-trained/EgoVLPv2_smallproj.pth) (for zero-shot evaluation) or [EgoVLPv2_EK-100](http://www.cis.jhu.edu/~shraman/EgoVLPv2/ckpts/EK-100_Finetuned/EK-100_finetune.pth) (for fine-tune evaluation) checkpoint, and update `load_checkpoint` in `configs/eval/epic.json` with its path.

```bash
python multinode_train_epic.py --save_dir eval_epic --config ./configs/eval/epic.json --print_freq 100
```

## 📊 Fine-tuned Checkpoints and Results on Charades-Ego

| Method | Eval | Checkpoint | mAP |
| :----: | :-----: | :--------: | :---------: |
| EgoVLPv2 | Zero-Shot | [EgoVLPv2_smallproj_epoch1](http://www.cis.jhu.edu/~shraman/EgoVLPv2/ckpts/Pre-trained/EgoVLPv2_smallproj_epoch1.pth) | 26.2 |
| EgoVLPv2 | Fine-tune | [EgoVLPv2_Charades-Ego](http://www.cis.jhu.edu/~shraman/EgoVLPv2/ckpts/Charades-Ego_Finetuned/Charades-Ego_finetune.pth) | 34.1 |


## 🎯 Fine-tuning on Charades-Ego

Download [EgoVLPv2_smallproj_epoch1](http://www.cis.jhu.edu/~shraman/EgoVLPv2/ckpts/Pre-trained/EgoVLPv2_smallproj_epoch1.pth) checkpoint, and update `load_checkpoint` in `configs/ft/charades.json` with its path.

```bash
python multinode_train_charades.py --save_dir finetune_charades --config ./configs/ft/charades.json --print_freq 100
```

For fine-tuning on multi-nodes, prepare a slurm script similar to `run.sh`. Since Charades-Ego has a small-sized training set, we use 32 frames for fine-tuning. Our Charades-Ego fine-tuning log can be found [here](http://www.cis.jhu.edu/~shraman/EgoVLPv2/logs/Charades-Ego_Finetune_log.txt).

## ⚓️ Zero-shot and Fine-tuned Checkpoint Evaluation on Charades-Ego (test-only)

Download [EgoVLPv2_smallproj_epoch1](http://www.cis.jhu.edu/~shraman/EgoVLPv2/ckpts/Pre-trained/EgoVLPv2_smallproj_epoch1.pth) (for zero-shot evaluation) or [EgoVLPv2_Charades-Ego](http://www.cis.jhu.edu/~shraman/EgoVLPv2/ckpts/Charades-Ego_Finetuned/Charades-Ego_finetune.pth) (for fine-tune evaluation) checkpoint, and update `load_checkpoint` in `configs/eval/charades.json` with its path.

```bash
python multinode_train_charades.py --save_dir eval_charades --config ./configs/eval/charades.json --print_freq 100
```


## ⚙️ Feature extraction on EgoMQ
We use PyTorch’s DataParallel (DP) for EgoMQ feature extraction. Download [EgoVLPv2](http://www.cis.jhu.edu/~shraman/EgoVLPv2/ckpts/Pre-trained/EgoVLPv2.pth) checkpoint, and update `load_checkpoint` in `configs/eval/mq.json` with proper pre-trained checkpoint path.
- We provide our pre-extracted MQ features [here](). 
The MQ features can be extracted as:
```bash
python test_mq.py --config configs/eval/mq.json --save_dir features_mq --split train --cuda_base cuda:0 --device_ids 0,1,2,3,4,5,6,7
python test_mq.py --config configs/eval/mq.json --save_dir features_mq --split test --cuda_base cuda:0 --device_ids 0,1,2,3,4,5,6,7
python test_mq.py --config configs/eval/mq.json --save_dir features_mq --split val --cuda_base cuda:0 --device_ids 0,1,2,3,4,5,6,7
# Note that the train, test and validation features should be under same root directory. 
```
These pre-extracted features are used for EgoMQ head-tuning, which is shown [here](https://github.com/ShramanPramanick/EgoVLPv2/tree/main/EgoMQ#-fine-tuning-on-egomq). 

## 🙏 Acknowledgement
The pre-training pipeline partially uses [EgoVLP](https://github.com/showlab/EgoVLP/tree/f3e8895c7a1a691bc7fb0c07618c3be0015887eb) and [FIBER](https://github.com/microsoft/FIBER) implementations. EK-100 dataloader partially uses [LAVILA](https://github.com/facebookresearch/LaViLa) codebase.
