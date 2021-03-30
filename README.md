## Curiosity-driven Exploration by Self-supervised Prediction ##
#### In ICML 2017 [[Project Website]](http://pathak22.github.io/noreward-rl/) [[Demo Video]](http://pathak22.github.io/noreward-rl/index.html#demoVideo)

[Deepak Pathak](https://people.eecs.berkeley.edu/~pathak/), [Pulkit Agrawal](https://people.eecs.berkeley.edu/~pulkitag/), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros/), [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/)<br/>
University of California, Berkeley<br/>

<img src="images/mario1.gif" width="300">    <img src="images/vizdoom.gif" width="351">

This is a tensorflow based implementation for our [ICML 2017 paper on curiosity-driven exploration for reinforcement learning](http://pathak22.github.io/noreward-rl/). Idea is to train agent with intrinsic curiosity-based motivation (ICM) when external rewards from environment are sparse. Surprisingly, you can use ICM even when there are no rewards available from the environment, in which case, agent learns to explore only out of curiosity: 'RL without rewards'. If you find this work useful in your research, please cite:

    @inproceedings{pathakICMl17curiosity,
        Author = {Pathak, Deepak and Agrawal, Pulkit and
                  Efros, Alexei A. and Darrell, Trevor},
        Title = {Curiosity-driven Exploration by Self-supervised Prediction},
        Booktitle = {International Conference on Machine Learning ({ICML})},
        Year = {2017}
    }

### 1) Usage
This repo is a modified version of the original implementation.
Only doom envorinment is included.
Here we use A2C instead of A3C.
Also, the original doom envorinment is too old. I use ViZDoom instead.
I keep the original model structure, but remove a lot of flag about the experiment settings.

This version is tested on the below environment:
* Ubuntu 18.04
* Tensorflow==1.15
* vizdoom==1.1.8

To train the model.
```
python main.py
```


### 2) Other helpful pointers
- [Paper](https://pathak22.github.io/noreward-rl/resources/icml17.pdf)
- [Project Website](http://pathak22.github.io/noreward-rl/)
- [Demo Video](http://pathak22.github.io/noreward-rl/index.html#demoVideo)
- [Reddit Discussion](https://redd.it/6bc8ul)
- [Media Articles (New Scientist, MIT Tech Review and others)](http://pathak22.github.io/noreward-rl/index.html#media)

### 3) Acknowledgement
Vanilla A3C code is based on the open source implementation of [universe-starter-agent](https://github.com/openai/universe-starter-agent).
