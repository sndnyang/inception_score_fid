# Compute the Inception Score and FID using TF and PyTorch

I give the demo in jupyter notebook files to show the usage and difference. PyTorch code is faster. FID values are also different.

The difference on cifar10:

| dataset      | TF      | PyTorch | gap |
|--------------|---------|---------|-----|
| Train        | 0.08768 | 0.20264 |     |
| Test         | 2.20365 | 5.12468 |     |
| Fake samples | 16.0426 | 9.87148 |     |




Tensorflow: please check demo.py, or eval_is_fid_tf.ipynb, using inception.py and fid.py

```python
# [n, 32, 32, 3]
fid = get_fid_score(test_images, test_ims)
score, std = get_inception_score(test_images, splits=splits)
```

PyTorch: please check eval_is_fid_pytorch.ipynb, using fid_score.py

```python
# [n, 3, 32, 32]
calculate_fid_score(feed_imgs2, test_imgs)
No Inception Score
```

### 

A set of generated images (10000 images) [npy file](https://1drv.ms/u/s!AgCFFlwzHuH8nHNpfwwzZRP0tDes?e=xtzG2n)


## Tensorflow Code

The code is based on the repo released by [IGEBM](https://arxiv.org/pdf/1903.08689.pdf)

It works fine, but  

1. sometimes TF is hard to run on GPU (CPU is so slow). Suggest to use conda [install tensorflow in conda](https://yann-leguilly.gitlab.io/post/2019-10-08-tensorflow-and-cuda/)
2. or occupies all memory of GPU so that you can't trace your training process. I have added the config in the code [inception score](https://github.com/sndnyang/inception_score_fid/blob/master/inception.py#L25], [FID](https://github.com/sndnyang/inception_score_fid/blob/master/fid.py#L25)  but I can't guarantee.

As a refer:
[Inception Score Row 231](https://github.com/openai/ebm_code_release/blob/master/test_inception.py#L231) 
and [FID Row 39](https://github.com/openai/ebm_code_release/blob/master/fid.py#L39)

## PyTorch Code 

mainly based on  https://github.com/mseitzer/pytorch-fid  this one computes based on the files stored in disk (two directorys)

As a refer:
[FID pytorch](https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py)
