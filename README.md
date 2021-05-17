Tensorflow2 implementation of various music auto-tagging models
==

<b>Evaluation of CNN-based Automatic Music Tagging Models</b> 

Minz Won, Andres Ferraro, Dmitry Bogdanov, and Xavier Serra

SMC, 2020

Reference
--

* [arxiv](https://arxiv.org/abs/2006.00751)

* [pytorch](https://github.com/minzwon/sota-music-tagging-models)


Available Models
--
* Sample-level CNN : Sample-level Deep Convolutional Neural Networks for Music Auto-tagging using Raw Waveforms, Lee et al., 2017 [pdf](https://mac.kaist.ac.kr/pubs/LeeParkKimNam-smc2017.pdf)
* Sample-level CNN + Squeeze-and-excitation : Sample-level CNN Architectures for Music Auto-tagging Using Raw Waveforms, Kim et al., 2018 [arxiv](https://arxiv.org/abs/1710.10451), [code](https://github.com/tae-jun/resemul)
* Music-SincNet : Music Auto-tagging with Learning Filter Banks, Lee and Shin, 2020 [pdf](https://github.com/jaehwlee/music-sincnet/files/5760095/2020KSC_.pdf)
* Harmonic-CNN : Data-driven Harmonic filters for Audio Representation Learning, Won et al., 2020 [pdf](https://ccrma.stanford.edu/~urinieto/MARL/publications/ICASSP2020_Won.pdf), [code](https://github.com/minzwon/data-driven-harmonic-filters)

Upcoming Models
--
* Self-attention


Usage
--
* Requirements
<pre>
<code>
conda env create -n {ENV_NAME} --file environment.yaml
conda activate {ENV_NAME}
</code>
</pre>

* Preprocessing
<pre>
<code>
python -u preprocess.py run ../dataset
python -u split.py run ../dataset
</code>
</pre>

* Training
<pre>
<code>
python main.py
</code>
</pre>

* Options
<pre>
<code>
'--gpu', type=str, default='0'
'--encoder_type', type=str, default='HC', choices=['HC', 'MS', 'SC']
'--block', type=str, default='rese', choices=['basic', 'se', 'res', 'rese']
</code>
</pre>
