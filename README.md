## [Earthquake Signal Detection Using a Multi-Scale Feature Fusion Network with Hybrid Attention Mechanism](https://academic.oup.com/gji/article/240/2/988/7912583)
<div align="center">
  <a href="https://github.com/cuiyang512" target="_blank">Yang Cui<sup>1</sup></a> &emsp;
  <a href="https://dqwl.yangtzeu.edu.cn/info/1171/5558.htm" target="_blank">Min Bai<sup>1 ,†</sup></a> &emsp;
  <a href="https://dqwl.yangtzeu.edu.cn/info/1169/5559.htm" target="_blank">Juan Wu<sup>1</sup></a> &emsp;
  <a href="https://github.com/chenyk1990" target="_blank">Yangkang Chen<sup>2</sup></a>
</div>
<div align="center">
  <sup>1</sup>College of Geophsyics and Petroleum Resources, Yangtze University<br>
  <sup>2</sup>The University of Texas at Austin 
</div>

<div align="center">
  <img src="https://img.shields.io/github/stars/cuiyang512/Multi-task-EQDetection" alt="GitHub Stars" />
  <img src="https://img.shields.io/github/forks/cuiyang512/Multi-task-EQDetection" alt="GitHub Forks" />
  <img src="https://img.shields.io/github/repo-size/cuiyang512/Multi-task-EQDetection" alt="Repo Size" />
  <img src="https://img.shields.io/github/last-commit/cuiyang512/Multi-task-EQDetection" alt="Last Commit" />
  <img src="https://img.shields.io/badge/language-Jupyter%20Notebook-%233572A5" alt="Language" />
</div>

## Overview
Signal and noise classification can add an extra level of constraint for earthquake phase picking by pinpointing the signal waveforms from continuous seismic data for more accurate arrival picking. However, the continuously increasing data collected by worldwide stations exceeds the ability of manual analysis. Moreover, manual earthquake data analysis depends on seismologists' expert knowledge, resulting in inconsistent analysis results. To address this, we proposed a generalized deep learning (DL) network architecture to discriminate earthquake signal and noise waveforms. The proposed DL framework is a novel architecture comprising a feature extractor, a classifier, and two hybrid attention modules. It utilizes different kernel sizes for more detailed feature extraction, and the hybrid attention mechanism module can guide the network to focus more on the waveform characteristics. To illustrate the power of the proposed DL network, we applied it to classify the earthquake signal and noise of the 3-C Texas Earthquake Dataset (TXED). The results demonstrate that the accuracy of the proposed method in the testing set reaches 99.83\%. We further utilize the transfer learning strategy to demonstrate the transferability of the proposed network with the Stanford earthquake dataset, showing an encouraging classification accuracy of 95.03\%. Additionally, we conducted an additional experiment on arrival picking by integrating decoder blocks into the classification network, which achieves remarkable P- and S-wave arrival picking accuracy.

![Network architecture](figs/Network.png)

## The testing data
Due to the limitation of Github, please refer to the following link to download the testing data:  https://drive.google.com/drive/folders/14kaFvHdMTJc2XL7Wl498F17UmfaidEfx?usp=drive_link

After downloading the testing data and TXED, please put them into the "data" folder

## TXED data
In this package, we used an excellent open-source earthquake data for AI -- TXED, please refer to the following link for more details: https://github.com/chenyk1990/txed 

Addtionally, please use the following link to download the TXED: https://drive.google.com/drive/folders/1WXVB8ytNB4bOaZ97oq6OmMRyAEg95trp?usp=sharing

## Reference
    Cui, Y., M. Bai, J. Wu, and Y. Chen. "Earthquake signal detection using a multi-scale feature fusion network with hybrid attention mechanism." Geophysical Journal International (2024): ggae423.

BibTex

    @Article{EQdetection,
      title={Earthquake signal detection using a multi-scale feature fusion network with hybrid attention mechanism},
      author={Cui, Y and Bai, M and Wu, J and Chen, Y},
      journal={Geophysical Journal International},
      pages={ggae423},
      year={2024},
      publisher={Oxford University Press}
    }

## Install 
For set up the environment and install the dependency packages, please run the following script:
    
    conda create -n mEQd python=3.11.7
    conda activate mEQd
    conda install ipython notebook
    pip install tensorflow==2.15.0, keras==2.15.0, h5py==3.10.0, scikit-learn==1.4.0, seaborn==0.13.2, matplotlib==3.8.4, scipy==1.11.4

If you want to run with GPUs, please run the following script to install the dependency package:

    conda create -n mEQd python=3.11.7
    conda activate mEQd
    conda install ipython notebook
    pip install tensorflow==2.15.0, keras==2.15.0, h5py==3.10.0, scikit-learn==1.4.0, seaborn==0.13.2, matplotlib==3.8.4, scipy==1.11.4
    conda install -c anaconda cudnn==8.2.1

## Development

    The development team welcomes voluntary contributions from any open-source enthusiast. 
    If you want to make contribution to this project, feel free to contact the development team. 
    
## Contact

    Regarding any questions, bugs, developments, collaborations, please contact  
    Yang Cui
    yang.cui512@gmail.com
