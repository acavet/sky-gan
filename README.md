# Deep convolutional generative adversarial network trained on sky images

Unfiltered example training images:
![](/assets/example-inputss.png)

Unfiltered example results:
![](/assets/example-outputs.png)

Generator architecture: 
![](https://www.researchgate.net/publication/331282441/figure/fig3/AS:729118295478273@1550846756282/Deep-convolutional-generative-adversarial-networks-DCGAN-for-generative-model-of-BF-NSP.png)

- Architecture modified from [this 2016 paper](https://arxiv.org/pdf/1511.06434.pdf) introducing DCGANs
- 16957 training images scraped from [r/SkyPorn](https://www.reddit.com/r/skyporn/) using Pushshift
- Trained and deployed on local CPU (code for training on AWS Sagemaker GPUs included)
- GAN code inspired by [bvshyam/facegeneration_gan_sagemaker](https://github.com/bvshyam/facegeneration_gan_sagemaker) and [vjrahil/Face-Generator](https://github.com/vjrahil/Face-Generator/blob/master/dlnd_face_generation.ipynb)