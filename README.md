# <Iterative Deep Homography Estimation>

<img src='https://github.com/imdumpl78/IHN/blob/main/demo.gif' width=600 >
    
## Description
- This is the open source implementation of the CVPR2022 paper "Iterative Deep Homography Estimation"  
Si-Yuan Cao, Jianxin Hu, ZeHua Sheng, Hui-Liang Shen  
*Zhejiang University, Hangzhou, China*
- [Paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Cao_Iterative_Deep_Homography_Estimation_CVPR_2022_paper.html)
- An improved version of IHN will be released soon.
    

## Table of Contents
- [Requirements](#requirements)
- [Usage](#usage)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)
- [Acknowledgement](#acknowledgement)

## Requirements
- Create a new anaconda environment and install all required packages before running the code.
```
conda create --name ihn
conda activate ihn
pip install requirements.txt
```

## Usage
The configurations are made in file .vscode/launch.json.   
Please notice the location of the datasets in file datasets_4cor_img.py (for mscoco and spid) and datasets.py(for googlemap and googleearth).  
Please note that the json file is only tested in vscode terminal.
    
## License
This project is released under the Apache 2.0 license.

## Citation
```
@InProceedings{Cao_2022_CVPR,
    author    = {Cao, Si-Yuan and Hu, Jianxin and Sheng, Zehua and Shen, Hui-Liang},
    title     = {Iterative Deep Homography Estimation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {1879-1888}
}
```
## Contact
- Email: karlcao@hotmail.com
    
## Acknowledgement
- This work is mainly based on [RAFT](https://github.com/princeton-vl/RAFT) and [SCV](https://github.com/zacjiang/SCV), we thank the authors for the contribution.
