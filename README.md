# Sorbet
Code for Sorbet model

## Quick start
run `bash run_glue.sh $TaskName$ $Timestep$`

Example: `bash run_glue.sh SST-2 8`

Specifically, `Timestep` and `Tbias` in the SNN model file need to be specified according to the timestep you are using.

## Pretrained full precision ANN weight for each dataset
Can be directly downloaded:
- MNLI: https://huggingface.co/textattack/bert-base-uncased-MNLI
- QQP:	https://huggingface.co/textattack/bert-base-uncased-QQP
- QNLI:	https://huggingface.co/textattack/bert-base-uncased-QNLI
- SST-2:	https://huggingface.co/textattack/bert-base-uncased-SST-2
- STS-B:	https://huggingface.co/textattack/bert-base-uncased-STS-B
- MRPC:	https://huggingface.co/textattack/bert-base-uncased-MRPC
- RTE:	https://huggingface.co/textattack/bert-base-uncased-RTE

## Trained Sorbet model on SST-2 dataset
We provide a set of weights to evaluate our Sorbet model. Please also change the path in the script accordingly.
https://drive.google.com/file/d/1tPBCvqxVH8JsaeBBhbQLUNARX6tGuqXb/view?usp=share_link



## Training hyperparameters

| Dataset | Max Seq Length | Batch Size | Learning Rate |
| ------- | -------------- | ---------- | ------------- |
| mnli    | 128            | 120        | 1e-5          |
| mrpc    | 128            | 40         | 1e-6          |
| sst-2   | 64             | 100        | 1e-6          |
| sts-b   | 128            | 30         | 5e-7          |
| qqp     | 128            | 100        | 1e-5          |
| qnli    | 128            | 80         | 1e-6          |
| rte     | 128            | 10         | 5e-6          |


Note: All the distillation steps are run for 100 epochs with an early stop setting.

## Clarification for Assumption B.1 in the paper
We show the distribution of the L1 norm of the inputs to the normalization layers by randomly sampling 3 batches and plotting the figures. The distribution supports our assumption that B.1 is mild regarding the L1 norm value. It's important to note that, due to the high feature dimension (768) in Sorbet, these values can be quite large. However, even with a reduction to 128 dimensions, the assumption still holds.


![Example Image](./figures/B1figure1.png)
![Example Image](./figures/B1figure2.png)
![Example Image](./figures/B1figure3.png)
