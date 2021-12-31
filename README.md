


------------------


### RNN Architectures

#### Structure

* **Bi-directional RNN [[Paper](http://www.di.ufpe.br/~fnj/RNA/bibliografia/BRNN.pdf)]**
* **Multi-dimensional RNN [[Paper](http://arxiv.org/pdf/0705.2011.pdf)]**
* **GFRNN [[Paper-arXiv](http://arxiv.org/pdf/1502.02367)] [[Paper-ICML](http://jmlr.org/proceedings/papers/v37/chung15.pdf)]** [[Supplementary](http://jmlr.org/proceedings/papers/v37/chung15-supp.pdf)]
* **Tree-Structured RNNs**  [[Paper](http://arxiv.org/pdf/1503.00075)]  [[Paper](http://arxiv.org/pdf/1506.04834)]
* **Grid LSTM [[Paper](http://arxiv.org/pdf/1507.01526)] [[Code](https://github.com/coreylynch/grid-lstm)]**
* **Segmental RNN [[Paper](http://arxiv.org/pdf/1511.06018v2.pdf)]**
* **Seq2seq for Sets [[Paper](http://arxiv.org/pdf/1511.06391v4.pdf)]**
* **Hierarchical Recurrent Neural Networks [[Paper](http://arxiv.org/abs/1609.01704)]**


#### Memory

* **LSTM [[Paper](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)] [Paper](https://arxiv.org/pdf/1611.05104.pdf)**
* **GRU (Gated Recurrent Unit) [[Paper](http://arxiv.org/pdf/1406.1078.pdf)]**
* **NTM [[Paper](http://arxiv.org/pdf/1410.5401)] [[Paper](https://arxiv.org/pdf/1703.03906.pdf)]** 
* **Neural GPU [[Paper](http://arxiv.org/pdf/1511.08228.pdf)]**
* **Memory Network [[Paper](http://arxiv.org/pdf/1410.3916)]**
* **Pointer Network [[Paper](http://arxiv.org/pdf/1506.03134)]**
* **Deep Attention Recurrent Q-Network [[Paper](http://arxiv.org/abs/1512.01693)]**
* **Dynamic Memory Networks [[Paper](http://arxiv.org/abs/1506.07285)]**

----------------
##### Timeline of RNNs
![rnn_timeline](https://github.com/gopala-kr/recurrent-nn/blob/master/res/rnn_timeline.PNG)

--------------
##### Comparision of LSTM networks
![lstm](https://github.com/gopala-kr/recurrent-nn/blob/master/res/lstm.PNG)

-----------
##### Comparision of RNN networks

![rnn](https://github.com/gopala-kr/recurrent-nn/blob/master/res/rnn.PNG)

----------------

### NLP

#### Basic Embedding Model

- [NNLM(Neural Network Language Model)](https://github.com/graykode/nlp-tutorial/tree/master/1-1.NNLM) - **Predict Next Word**
[[Paper(2003)](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)]
[[NNLM_Tensor.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/1-1.NNLM/NNLM_Tensor.ipynb)] [[NNLM_Torch.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/1-1.NNLM/NNLM_Torch.ipynb)]
- [Word2Vec(Skip-gram)](https://github.com/graykode/nlp-tutorial/tree/master/1-2.Word2Vec) - **Embedding Words and Show Graph**
[[Distributed Representations of Words and Phrases
    and their Compositionality(2013)](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)]
[[Word2Vec_Tensor(NCE_loss).ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/1-2.Word2Vec/Word2Vec_Skipgram_Tensor(NCE_loss).ipynb)] [[Word2Vec_Tensor(Softmax).ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/1-2.Word2Vec/Word2Vec_Skipgram_Tensor(Softmax).ipynb)]
[[Word2Vec_Torch(Softmax).ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/1-2.Word2Vec/Word2Vec_Skipgram_Torch(Softmax).ipynb)]
- [FastText(Application Level)](https://github.com/graykode/nlp-tutorial/tree/master/1-3.FastText) - **Sentence Classification**
[[Bag of Tricks for Efficient Text Classification(2016)](https://arxiv.org/pdf/1607.01759.pdf)]
[[FastText.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/1-3.FastText/FastText.ipynb)]



#### CNN(Convolutional Neural Network)

- [TextCNN](https://github.com/graykode/nlp-tutorial/tree/master/2-1.TextCNN) - **Binary Sentiment Classification**
[[Paper(2014)](http://www.aclweb.org/anthology/D14-1181)]
[[Colab](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/2-1.TextCNN/TextCNN_Tensor.ipynb)] 
[[TextCNN_Torch.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/2-1.TextCNN/TextCNN_Torch.ipynb)
-  DCNN(Dynamic Convolutional Neural Network)



#### RNN(Recurrent Neural Network)

- [TextRNN](https://github.com/graykode/nlp-tutorial/tree/master/3-1.TextRNN) - **Predict Next Step**
[[Finding Structure in Time(1990)](http://psych.colorado.edu/~kimlab/Elman1990.pdf)]
[[TextRNN_Tensor.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/3-1.TextRNN/TextRNN_Tensor.ipynb)] [[TextRNN_Torch.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/3-1.TextRNN/TextRNN_Torch.ipynb)]
- [TextLSTM](https://github.com/graykode/nlp-tutorial/tree/master/3-2.TextLSTM) - **Autocomplete** [[LONG SHORT-TERM MEMORY(1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)]
 [[TextLSTM_Tensor.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/3-2.TextLSTM/TextLSTM_Tensor.ipynb)] [[TextLSTM_Torch.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/3-2.TextLSTM/TextLSTM_Torch.ipynb)]
- [Bi-LSTM](https://github.com/graykode/nlp-tutorial/tree/master/3-3.Bi-LSTM) - **Predict Next Word in Long Sentence**
[[Bi_LSTM_Tensor.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/3-3.Bi-LSTM/Bi_LSTM_Tensor.ipynb)] [[Bi_LSTM_Torch.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/3-3.Bi-LSTM/Bi_LSTM_Torch.ipynb)]



#### Attention Mechanism

- [Seq2Seq](https://github.com/graykode/nlp-tutorial/tree/master/4-1.Seq2Seq) - **Change Word**
[[Paper(2014)](https://arxiv.org/pdf/1406.1078.pdf)]
[[Colab](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/4-1.Seq2Seq/Seq2Seq_Tensor.ipynb)] [[Seq2Seq_Torch.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/4-1.Seq2Seq/Seq2Seq_Torch.ipynb)]
- [Seq2Seq with Attention](https://github.com/graykode/nlp-tutorial/tree/master/4-2.Seq2Seq(Attention)) - **Translate**
[[Paper(2014)](https://arxiv.org/abs/1409.0473)]
[[Colab](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/4-2.Seq2Seq(Attention)/Seq2Seq(Attention)_Tensor.ipynb)]  [[Seq2Seq(Attention)_Torch.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/4-2.Seq2Seq(Attention)/Seq2Seq(Attention)_Torch.ipynb)]
- [Bi-LSTM with Attention](https://github.com/graykode/nlp-tutorial/tree/master/4-3.Bi-LSTM(Attention)) - **Binary Sentiment Classification**
[[Bi_LSTM(Attention)_Tensor.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/4-3.Bi-LSTM(Attention)/Bi_LSTM(Attention)_Tensor.ipynb)]  [[Bi_LSTM(Attention)_Torch.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/4-3.Bi-LSTM(Attention)/Bi_LSTM(Attention)_Torch.ipynb)]



#### Model based on Transformer

- [The Transformer](https://github.com/graykode/nlp-tutorial/tree/master/5-1.Transformer) - **Translate**
  [[Paper(2017)](https://arxiv.org/abs/1810.04805)]  [[Colab](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/5-1.Transformer/Transformer_Torch.ipynb)], [[Transformer(Greedy_decoder)_Torch.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/5-1.Transformer/Transformer(Greedy_decoder)_Torch.ipynb)] 
- [BERT](https://github.com/graykode/nlp-tutorial/tree/master/5-2.BERT) - **Classification Next Sentence & Predict Masked Tokens**
[[Paper(2018)](https://arxiv.org/abs/1810.04805)]  [[Colab](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/5-2.BERT/BERT_Torch.ipynb)]

------------------
--------------------
![1*ff_bprXLuTueAx7-5-MHew](https://miro.medium.com/max/1500/1*ff_bprXLuTueAx7-5-MHew.png)
--------
![FvQ12Yic_Iif2mxJB64bNw](https://miro.medium.com/max/1500/1*FvQ12Yic_Iif2mxJB64bNw.png)
-----------
![1*X1JSg2zYqD94Mp-MJRBsAw](https://miro.medium.com/max/1050/1*X1JSg2zYqD94Mp-MJRBsAw.png)
---------
![zCoB9_l5NXzlggQikrdxYg](https://miro.medium.com/max/1500/1*zCoB9_l5NXzlggQikrdxYg.png)

---------------
---------------------
#### NLP Pre-Trained Models

![PTM2](https://github.com/gopala-kr/language-models/blob/master/res/PTM2.PNG)

--------------------

#### Taxonomy of Pre-Trained Models (NLP)


![PTM1](https://github.com/gopala-kr/language-models/blob/master/res/PTM1.PNG)


--------------
![comp](https://github.com/gopala-kr/language-models/blob/master/res/comp.PNG)

--------------
#### Benchmarks

![Benchmarks](https://github.com/gopala-kr/language-models/blob/master/res/Benchmarks.PNG)

---------------
![PLMfamily](https://github.com/thunlp/PLMpapers/blob/master/PLMfamily.jpg)

----------------
![Compression](https://github.com/gopala-kr/language-models/blob/master/res/Compression.PNG)

---------------------------
![textmining](https://raw.githubusercontent.com/graykode/nlp-roadmap/master/img/textmining.png)

[source : graykode/nlp-roadmap](https://github.com/graykode/nlp-roadmap)

---------------

![nlp](https://raw.githubusercontent.com/graykode/nlp-roadmap/master/img/nlp.png)

[source : graykode/nlp-roadmap](https://github.com/graykode/nlp-roadmap)

----------------------------
#### References

- [NLP SOTA](https://paperswithcode.com/area/natural-language-processing)
- [rnn-surveys](https://github.com/gopala-kr/recurrent-nn/blob/master/rnn-surveys.md)
- [ref-implementations](https://github.com/gopala-kr/recurrent-nn/blob/master/ref-implementations.md)
- [language-modeling](https://github.com/gopala-kr/recurrent-nn/blob/master/language-modeling.md)
- [conversation-modeling](https://github.com/gopala-kr/recurrent-nn/blob/master/conversation-modeling.md)
- [machine-translation](https://github.com/gopala-kr/recurrent-nn/blob/master/machine-translation.md)
- [qa](https://github.com/gopala-kr/recurrent-nn/blob/master/qa.md)
- [speech-processing](https://github.com/gopala-kr/recurrent-nn/blob/master/speech-processing.md)
- [vision-nlp](https://github.com/gopala-kr/recurrent-nn/blob/master/vision-nlp.md)
- [rnn-vision](https://github.com/gopala-kr/recurrent-nn/blob/master/rnn-vision.md)
- [rnn-robot](https://github.com/gopala-kr/recurrent-nn/blob/master/rnn-robot.md)
- [turing-machines](https://github.com/gopala-kr/recurrent-nn/blob/master/turing-machines.md)
- [rnn-other](https://github.com/gopala-kr/recurrent-nn/blob/master/rnn-other.md)
- [BERT-related-papers](https://github.com/tomohideshibata/BERT-related-papers)
- [NiuTrans/ABigSurvey](https://github.com/NiuTrans/ABigSurvey)
-----------

- [A Survey: Time Travel in Deep Learning Space: An Introduction to Deep Learning Models and How Deep Learning Models Evolved from the Initial Ideas](https://arxiv.org/abs/1510.04781)
- [Survey on the attention based RNN model and its applications in computer vision](https://arxiv.org/pdf/1601.06823.pdf)
- [Automatic Description Generation from Images: A Survey of Models, Datasets, and Evaluation Measures](https://arxiv.org/pdf/1601.03896.pdf)
- [Neural Machine Translation and Sequence-to-sequence Models: A Tutorial](https://arxiv.org/pdf/1703.01619.pdf)
- [Best Practices for Applying Deep Learning to Novel Applications](https://arxiv.org/ftp/arxiv/papers/1704/1704.01568.pdf)
- [ParlAI: A Dialog Research Software Platform](https://arxiv.org/pdf/1705.06476.pdf)
- [Statistical Machine Translation](https://arxiv.org/pdf/1709.07809.pdf)
- [Adversarial Examples: Attacks and Defenses for Deep Learning](https://arxiv.org/pdf/1712.07107.pdf)
- [Deep Learning:
A Critical Appraisal ](https://arxiv.org/ftp/arxiv/papers/1801/1801.00631.pdf)
- [From Word to Sense Embeddings: A Survey on Vector Representations of Meaning](https://arxiv.org/abs/1805.04032v1)
- [Natural Language Processing for Information Extraction](https://arxiv.org/abs/1807.02383v1)
- [A Review of the Neural History of Natural Language Processing](http://ruder.io/a-review-of-the-recent-history-of-nlp/)
- [EMNLP 2018 Highlights: Inductive bias, cross-lingual learning, and more](http://ruder.io/emnlp-2018-highlights/)
- [A Survey of the Usages of Deep Learning in Natural Language Processing](https://arxiv.org/abs/1807.10854v1)
- [Adversarial Attacks and Defences: A Survey](https://arxiv.org/pdf/1810.00069v1.pdf)
- [Secure Deep Learning Engineering:
A Software Quality Assurance Perspective](https://arxiv.org/pdf/1810.04538v1.pdf)
- [Tackling Sequence to Sequence Mapping
Problems with Neural Networks](https://arxiv.org/pdf/1810.10802v1.pdf)
- [Security for Machine Learning-based Systems: Attacks and Challenges during Training and Inference](https://arxiv.org/abs/1811.01463v1)
- [Speech processing: recognition, synthesis + Survey on chatbot platforms and API's](https://github.com/gopala-kr/a-week-in-wild-ai/tree/master/03-speech-processing)
- [Fundamentals of Recurrent Neural Network (RNN) and Long Short-Term Memory (LSTM) Network](https://arxiv.org/abs/1808.03314v4)
- [Deep RNN Framework for Visual Sequential Applications](https://arxiv.org/abs/1811.09961v3)
- [EcoRNN: Efficient Computing of LSTM RNN Training on GPUs](https://arxiv.org/abs/1805.08899v4)
- [Training for 'Unstable' CNN Accelerator:A Case Study on FPGA](https://arxiv.org/abs/1812.01689v1)
- [Modular Mechanistic Networks: On Bridging Mechanistic and Phenomenological Models with Deep Neural Networks in Natural Language Processing](https://arxiv.org/abs/1807.09844v2)
- [Modeling Language Variation and Universals: A Survey on Typological Linguistics for Natural Language Processing](https://arxiv.org/abs/1807.00914v2)
- [Attention, please! A Critical Review of Neural Attention Models in Natural Language Processing](https://arxiv.org/abs/1902.02181v1)
- [Recent Trends in Deep Learning Based Natural Language Processing](https://arxiv.org/abs/1708.02709v8)
- [Quantifying Uncertainties in Natural Language Processing Tasks](https://arxiv.org/abs/1811.07253v1)
- [A Survey on Natural Language Processing for Fake News Detection](https://arxiv.org/abs/1811.00770v1)
- [Visualizing memorization in RNNs](https://distill.pub/2019/memorization-in-rnns/)
- [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf) | [Paper Explained](https://www.youtube.com/watch?v=SY5PvZrJhLE) | [openai/gpt-3](https://github.com/openai/gpt-3) | [OpenAI GPT-3](https://www.youtube.com/watch?v=_x9AwxfjxvE)
- [HuggingFace's Transformers: State-of-the-art Natural Language Processing](https://arxiv.org/pdf/1910.03771v4.pdf)
- [Trends in Integration of Vision and Language Research: A Survey of Tasks, Datasets, and Methods](https://arxiv.org/pdf/1907.09358.pdf)
- [A Survey of Evaluation Metrics Used for NLG Systems](https://arxiv.org/pdf/2008.12009v2.pdf)
- [NLPStatTest: A Toolkit for Comparing NLP System Performance](https://arxiv.org/pdf/2011.13231v1.pdf)
- [Robustness Gym: Unifying the NLP Evaluation Landscape](https://arxiv.org/pdf/2101.04840v1.pdf)
- [Exploring and Predicting Transferability across NLP Tasks](https://arxiv.org/pdf/2005.00770v2.pdf)
- [Neuron-level Interpretation of Deep NLP Models: A Survey](https://arxiv.org/pdf/2108.13138v1.pdf)
- [A Survey of Data Augmentation Approaches for NLP](https://arxiv.org/pdf/2105.03075v4.pdf)
- [A Short Survey of Pre-trained Language Models for Conversational AI-A NewAge in NLP](https://arxiv.org/pdf/2104.10810v1.pdf)
- [Language (Technology) is Power: A Critical Survey of "Bias" in NLP](https://arxiv.org/pdf/2005.14050v2.pdf)
- [Taxonomic survey of Hindi Language NLP systems](https://arxiv.org/ftp/arxiv/papers/2102/2102.00214.pdf)
- [An Introductory Survey on Attention Mechanisms in NLP Problems](https://arxiv.org/pdf/1811.05544v1.pdf)
- [Indian Legal NLP Benchmarks : A Survey](https://arxiv.org/pdf/2107.06056v1.pdf)
- [Post-hoc Interpretability for Neural NLP: A Survey](https://arxiv.org/pdf/2108.04840v2.pdf)
- [Explanation-Based Human Debugging of NLP Models: A Survey](https://arxiv.org/pdf/2104.15135v2.pdf)
- [Recent Advances in Natural Language Processing via Large Pre-Trained Language Models: A Survey](https://arxiv.org/pdf/2111.01243v1.pdf)

------------

- [linguistics](https://yandexdataschool.com/edu-process/linguistics)
- [Machine translation](https://yandexdataschool.com/edu-process/mt)
- [awesome-nlp](https://github.com/keon/awesome-nlp) 
- [awesome-bert](https://github.com/Jiakui/awesome-bert)
- [PLMpapers](https://github.com/thunlp/PLMpapers)
- [nlp-tutorial](https://github.com/graykode/nlp-tutorial)
- [nlp_tasks](https://github.com/Kyubyong/nlp_tasks) 
- [DeepNLP-models-Pytorch](https://github.com/DSKSD/DeepNLP-models-Pytorch) 
- [oxford.nlp.lectures](https://github.com/oxford-cs-deepnlp-2017/lectures) 
- [stanford.nlp.lectures](https://www.youtube.com/watch?v=OQQ-W_63UgQ&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6) 
- [nltk.org/book](http://www.nltk.org/book/) 
- [DL4NLP](https://github.com/andrewt3000/DL4NLP) 
- [cs388.utexas.nlp](https://www.cs.utexas.edu/~mooney/cs388/) 
- [nlp-datasets](https://github.com/karthikncode/nlp-datasets) 
- [DL-NLP-Readings](https://github.com/IsaacChanghau/DL-NLP-Readings) 
- [gt-nlp-class](https://github.com/jacobeisenstein/gt-nlp-class)
- [embedding-models](https://github.com/Hironsan/awesome-embedding-models)
- [Facebook: Advancing understanding at ACL 2017](https://research.fb.com/advancing-understanding-at-acl2017/)
- [Facebook: Visual reasoning and dialog](https://research.fb.com/visual-reasoning-and-dialog-towards-natural-language-conversations-about-visual-data/)
- [ilya_sutskever_phd_thesis](http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf)
- [Notes on state of the art techniques for language modeling](https://www.fast.ai/2017/08/25/language-modeling-sota/)
- [ASR 2017-18: lectures](https://www.inf.ed.ac.uk/teaching/courses/asr/lectures.html)
- [Sebastian Ruder](http://ruder.io/)
- [wer_are_we](https://github.com/syhw/wer_are_we)
- [NLP-progress](https://github.com/sebastianruder/NLP-progress)
- [NLP-Models-Tensorflow](https://github.com/huseinzol05/NLP-Models-Tensorflow)
- [graykode/nlp-roadmap](https://github.com/graykode/nlp-roadmap)
- [huggingface/nlp](https://github.com/huggingface) | [huggingface channel](https://www.youtube.com/channel/UCHlNU7kIZhRgSbhHvFoy72w)
- [bert-nlp](https://github.com/cedrickchee/awesome-bert-nlp)
- [The Future of Natural Language Processing](https://www.youtube.com/watch?v=G5lmya6eKtc)
- [paperswithcode/natural-language-processing](https://paperswithcode.com/area/natural-language-processing)
- [paperswithcode/speech](https://paperswithcode.com/area/speech)
- [nlp-methods](https://paperswithcode.com/methods/area/natural-language-processing)
- [practical-nlp/practical-nlp](https://github.com/practical-nlp/practical-nlp)
- [ICLR 2020: NLP Highlights](https://towardsdatascience.com/iclr-2020-nlp-highlights-511deb99b967)

------------------------

_**Maintainer**_

Gopala KR / @gopala-kr

----------------------
