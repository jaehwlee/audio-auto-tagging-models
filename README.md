TF2 implementation of Sample-level CNN Architectures for Music Auto-tagging Using Raw Waveforms
==
Taejun Kim, Jongpil Lee, and Juhan Nam

ICASSP, 2018

Reference 
--
[keras implementation](https://github.com/tae-jun/resemul)

[paper](https://arxiv.org/abs/1710.10451)

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
