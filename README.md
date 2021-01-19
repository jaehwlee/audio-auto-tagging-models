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
* Sample-level CNN + Squeeze-and-excitation : Sample-level CNN Architectures for Music Auto-tagging Using Raw Waveforms, Kim et al., 2018 [arxiv](https://arxiv.org/abs/1710.10451), [code](https://github.com/tae-jun/resemul)


Upcoming Models
--
* Self-attention
* Harmonic CNN


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
python train.py
</code>
</pre>
