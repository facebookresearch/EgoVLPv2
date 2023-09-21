## 📝 QFVS Data Preparation

QFVS dataset uses the same raw videos as the UT Egocentric (UT Ego) Dataset. We download the [UT Ego](https://vision.cs.utexas.edu/projects/egocentric_data/UT_Egocentric_Dataset.html) raw videos and [QFVS](https://openaccess.thecvf.com/content_cvpr_2017/papers/Sharghi_Query-Focused_Video_Summarization_CVPR_2017_paper.pdf) annotations. 
For quickstart and easy access for the users, we provide the preprocessed videos and annotations:

```bash
mkdir Datasets && cd Datasets
wget https://www.cis.jhu.edu/~shraman/EgoVLPv2/datasets/QFVS.tgz
tar -xvzf QFVS.tgz && rm QFVS.tgz
```

## 📊 Results

| Method | Video-1 | Video-2 | Video-3 | Video-4 | Average |
| :----: | :-----: | :--------: | :---------: | :------: | :------: | 
| EgoVLPv2 | 53.30 | 54.13 | 62.64 | 38.25 | 52.08 |

## 🎯 Fine-tuning on QFVS
Download [EgoVLPv2](http://www.cis.jhu.edu/~shraman/EgoVLPv2/ckpts/Pre-trained/EgoVLPv2.pth) checkpoint, and update `load_checkpoint` in `qfvs.json` with its path. QFVS dataset contains only 4 videos. Use 3 videos for training, and the rest for evaluation. Perform 4 different training runs to evaluate on 4 different videos. Change `device_ids` based on available GPUs.

```bash
mkdir multimodal_features/
python main.py --train_videos 2,3,4 --test_video 1 --cuda_base cuda:0 --device_ids 0,1,2,3
python main.py --train_videos 1,3,4 --test_video 2 --cuda_base cuda:0 --device_ids 0,1,2,3
python main.py --train_videos 1,2,4 --test_video 3 --cuda_base cuda:0 --device_ids 0,1,2,3
python main.py --train_videos 1,2,3 --test_video 4 --cuda_base cuda:0 --device_ids 0,1,2,3
```
Due to the tiny training set, the results on the QFVS dataset significantly vary across different runs. We have performed multiple runs and reported the best results.  

## 🙏 Acknowledgement
QFVS implementation partially uses [VASNet](https://github.com/ok1zjf/VASNet) codebase. 
