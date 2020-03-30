

<h1 align="center">Beyond The FRONTIER</h1>

<p align="center">
    <a href="https://www.tensorflow.org/">
        <img src="https://img.shields.io/badge/Tensorflow-1.13-green" alt="Vue2.0">
    </a>
    <a href="https://github.com/CuiJiali-CV/">
        <img src="https://img.shields.io/badge/Author-JialiCui-blueviolet" alt="Author">
    </a>
    <a href="https://github.com/CuiJiali-CV/">
        <img src="https://img.shields.io/badge/Email-cuijiali961224@gmail.com-blueviolet" alt="Author">
    </a>
    <a href="https://www.stevens.edu/">
        <img src="https://img.shields.io/badge/College-SIT-green" alt="Vue2.0">
    </a>
</p>

[Paper Here](https://arxiv.org/pdf/1606.08571.pdf)
<br /><br />
## BackGround

* The code works well both on Generation and Reconstruction on Mnist, and this is what I use for other project.

<br /><br />
## Quick Start

* Run train.py and the result will be stored in output
* The hyper params are already set up in train.py. 
    * theta is the param in loss. Smaller theta often means less difference allowed between true image and reproduce.<br />
    * delta is the param in langevin dynamic. it is the param that means how far for one time langevin dynamic process can walk for one step. Normally, in warm start, I won't suggest set this param too big. However, it is really interesting to see how the theta and delta working together on the model.<br />
    
* The number of trainning images can be set up in loadData.py. Just simply change num to any number of images you want to train
<br /><br />
## Version of Installment
#### Tensorflow 1.13.1
#### Numpy 1.18.2
#### Python 3.6.9  
<br />

## Structure of Network  
* In fact, the leaky ReLu should switch place with BN, but that's it, LOL.
* the code is correct, so don't worry about it.
* And this structure is the same as the Generator of GAN, [My GAN Implement](https://github.com/CuiJiali-CV/GAN)

### Generator
 ![Image text](https://github.com/CuiJiali-CV/cGAN/raw/master/Generator.png)

## Results
<br />

### Reconstruction

   the reproduce must be good in ABP<br />
 ![Image text](https://github.com/CuiJiali-CV/ABP/raw/master/Reconstruction.png)

#### Generation
 ![Image text](https://github.com/CuiJiali-CV/ABP/raw/master/Reconstruction.png)
<br /><br />
## Author

```javascript
var iD = {
  name  : "Jiali Cui",
  
  bachelor: "Harbin Institue of Technology",
  master : "Stevens Institute of Technology",
  
  Interested: "CV, ML"
}
```
