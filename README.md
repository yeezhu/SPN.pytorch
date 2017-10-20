# PyTorch implementation of SPN

Soft Proposal Networks for Weakly Supervised Object Localization, ICCV 2017.

[[Project Page]](http://yzhu.work/spn) [[Paper]](https://arxiv.org/pdf/1709.01829) [[Supp]](http://yzhu.work/pdffiles/SPN_Supp.pdf) 

[[Torch code]](https://github.com/ZhouYanzhao/SPN)  
[Caffe code] comming soon...

### Requirements
* Python3.5
* PyTorch: `conda install pytorch torchvision -c soumith`
* Packages: torch, [torchnet](https://github.com/pytorch/tnt), numpy, tqdm, 

### Usage

1. Clone the SPN repository: 
    ```bash
    git clone https://github.com/yeezhu/SPN.pytorch.git
    ```

2. Install SPN: 
    ```bash
    cd SPN.pytorch/spnlib
    bash make.sh
    ```

3. Run the demo (implemented based on [wildcat.pytorch](https://github.com/durandtibo/wildcat.pytorch)): 
    ```bash
    cd SPN.pytorch/demo
    bash runme.sh
    ```

## Citation 
If you use the code in your research, please cite:
```bibtex
@INPROCEEDINGS{Zhu2017SPN,
    author = {Zhu, Yi and Zhou, Yanzhao and Ye, Qixiang and Qiu, Qiang and Jiao, Jianbin},
    title = {Soft Proposal Networks for Weakly Supervised Object Localization},
    booktitle = {ICCV},
    year = {2017}
}
```
