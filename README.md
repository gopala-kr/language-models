


--------------------------

* basic LSTM networks(bi-directional and dynamic rnn)
* Kronecker Highway RNN
* Delta RNN
* Highway Networks
* Recurrent Highway Networks
* Multiplicative Integration Within RNNs
* Recurrent Dropout
* Layer Normalization
* Layer Normalization & Multiplicative Integration
* LSTM With Multiple Memory Arrays
* Minimal Gated Unit RNN
* Residual Connections Within Stacked RNNs
* GRU Mutants
* Weight Tying

------------------
### Surveys
* [Deep Learning](http://www.nature.com/nature/journal/v521/n7553/pdf/nature14539.pdf), Nature 2015
* [LSTM: A Search Space Odyssey](http://arxiv.org/pdf/1503.04069), arXiv:1503.04069
* [A Critical Review of Recurrent Neural Networks for Sequence Learning](http://arxiv.org/pdf/1506.00019), arXiv:1506.00019
* [Visualizing and Understanding Recurrent Networks](http://arxiv.org/pdf/1506.02078), arXiv:1506.02078
* [An Empirical Exploration of Recurrent Network Architectures](http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf), ICML, 2015.
- Recent Advances in Recurrent Neural Networks. 2018. [[arXiv](https://arxiv.org/abs/1801.01078v3)]
- From Nodes to Networks: Evolving Recurrent Neural Networks. 2018. [[arXiv](https://arxiv.org/abs/1803.04439v2)]
- The History Began from AlexNet: A Comprehensive Survey on Deep Learning Approaches. 2018. [[arXiv](https://arxiv.org/abs/1803.01164v1)]
- [Natural Language Processing: State of The Art, Current Trends and Challenges](https://arxiv.org/ftp/arxiv/papers/1708/1708.05148.pdf)
- [Recent Trends in Deep Learning Based
Natural Language Processing](https://arxiv.org/pdf/1708.02709v5.pdf)

### Architectures

#### Structure

* **Bi-directional RNN [[Paper](http://www.di.ufpe.br/~fnj/RNA/bibliografia/BRNN.pdf)]**
  * *Bidirectional Recurrent Neural Networks*, Trans. on Signal Processing 1997
* **Multi-dimensional RNN [[Paper](http://arxiv.org/pdf/0705.2011.pdf)]**
  * *Multi-Dimensional Recurrent Neural Networks*, ICANN 2007
* **GFRNN [[Paper-arXiv](http://arxiv.org/pdf/1502.02367)] [[Paper-ICML](http://jmlr.org/proceedings/papers/v37/chung15.pdf)]** [[Supplementary](http://jmlr.org/proceedings/papers/v37/chung15-supp.pdf)]
  * *Gated Feedback Recurrent Neural Networks*, arXiv:1502.02367 / ICML 2015
* **Tree-Structured RNNs**
  * *Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks*, arXiv:1503.00075 / ACL 2015 [[Paper](http://arxiv.org/pdf/1503.00075)]
  * *Tree-structured composition in neural networks without tree-structured architectures*, arXiv:1506.04834 [[Paper](http://arxiv.org/pdf/1506.04834)]
* **Grid LSTM [[Paper](http://arxiv.org/pdf/1507.01526)] [[Code](https://github.com/coreylynch/grid-lstm)]**
  * *Grid Long Short-Term Memory*, arXiv:1507.01526
* **Segmental RNN [[Paper](http://arxiv.org/pdf/1511.06018v2.pdf)]**
  * "Segmental Recurrent Neural Networks", ICLR 2016.
* **Seq2seq for Sets [[Paper](http://arxiv.org/pdf/1511.06391v4.pdf)]**
  * "Order Matters: Sequence to sequence for sets", ICLR 2016.
* **Hierarchical Recurrent Neural Networks [[Paper](http://arxiv.org/abs/1609.01704)]**
  * "Hierarchical Multiscale Recurrent Neural Networks", arXiv:1609.01704

#### Memory

* **LSTM [[Paper](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)]**
  * *Long Short-Term Memory*, Neural Computation 1997
* **GRU (Gated Recurrent Unit) [[Paper](http://arxiv.org/pdf/1406.1078.pdf)]**
  * *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation*, arXiv:1406.1078 / EMNLP 2014
* **NTM [[Paper](http://arxiv.org/pdf/1410.5401)]**
  * *Neural Turing Machines,* arXiv preprint arXiv:1410.5401
* **Neural GPU [[Paper](http://arxiv.org/pdf/1511.08228.pdf)]**
  * arXiv:1511.08228 / ICML 2016 (under review)
* **Memory Network [[Paper](http://arxiv.org/pdf/1410.3916)]**
  * *Memory Networks,* arXiv:1410.3916
* **Pointer Network [[Paper](http://arxiv.org/pdf/1506.03134)]**
  * *Pointer Networks*, arXiv:1506.03134 / NIPS 2015
* **Deep Attention Recurrent Q-Network [[Paper](http://arxiv.org/abs/1512.01693)]**
  *  *Deep Attention Recurrent Q-Network* , arXiv:1512.01693
* **Dynamic Memory Networks [[Paper](http://arxiv.org/abs/1506.07285)]**
  * "Ask Me Anything: Dynamic Memory Networks for Natural Language Processing", arXiv:1506.07285


## Applications

### Natural Language Processing

#### Language Modeling
*  "Honza" Cernocky, Sanjeev Khudanpur, *Recurrent Neural Network based Language Model*, Interspeech 2010 [[Paper](http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)]
* "Honza" Cernocky, Sanjeev Khudanpur, *Extensions of Recurrent Neural Network Language Model*, ICASSP 2011 [[Paper](http://www.fit.vutbr.cz/research/groups/speech/publi/2011/mikolov_icassp2011_5528.pdf)]
* *Recurrent Neural Network based Language Modeling in Meeting Recognition*, Interspeech 2011 [[Paper](http://www.fit.vutbr.cz/~imikolov/rnnlm/ApplicationOfRNNinMeetingRecognition_IS2011.pdf)]
* *A Hierarchical Neural Autoencoder for Paragraphs and Documents*, ACL 2015 [[Paper](http://arxiv.org/pdf/1506.01057)], [[Code](https://github.com/jiweil/Hierarchical-Neural-Autoencoder)]
*  *Skip-Thought Vectors*, arXiv:1506.06726 / NIPS 2015 [[Paper](http://arxiv.org/pdf/1506.06726.pdf)]
* *Character-Aware Neural Language Models*, arXiv:1508.06615 [[Paper](http://arxiv.org/pdf/1508.06615)]
* *Tree Recurrent Neural Networks with Application to Language Modeling*, arXiv:1511.00060 [[Paper](http://arxiv.org/pdf/1511.00060.pdf)]
* *The Goldilocks Principle: Reading children's books with explicit memory representations*, arXiv:1511.0230 [[Paper](http://arxiv.org/pdf/1511.02301.pdf)]


#### Speech Recognition
* *Deep Neural Networks for Acoustic Modeling in Speech Recognition*, IEEE Signam Processing Magazine 2012 [[Paper](http://cs224d.stanford.edu/papers/maas_paper.pdf)]
* *Speech Recognition with Deep Recurrent Neural Networks*, arXiv:1303.5778 / ICASSP 2013 [[Paper](http://www.cs.toronto.edu/~fritz/absps/RNN13.pdf)]
* *Attention-Based Models for Speech Recognition*, arXiv:1506.07503 / NIPS 2015 [[Paper](http://arxiv.org/pdf/1506.07503)]
* *Fast and Accurate Recurrent Neural Network Acoustic Models for Speech Recognition*, arXiv:1507.06947 2015 [[Paper](http://arxiv.org/pdf/1507.06947v1.pdf)].

#### Machine Translation
* Oxford [[Paper](http://www.nal.ai/papers/kalchbrennerblunsom_emnlp13)]
  * *Recurrent Continuous Translation Models*, EMNLP 2013
* Univ. Montreal
  * *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation*, arXiv:1406.1078 / EMNLP 2014 [[Paper](http://arxiv.org/pdf/1406.1078)]
  * *On the Properties of Neural Machine Translation: Encoder-Decoder Approaches*, SSST-8 2014 [[Paper](http://www.aclweb.org/anthology/W14-4012)]
  * *Overcoming the Curse of Sentence Length for Neural Machine Translation using Automatic Segmentation*, SSST-8 2014
  * *Neural Machine Translation by Jointly Learning to Align and Translate*, arXiv:1409.0473 / ICLR 2015 [[Paper](http://arxiv.org/pdf/1409.0473)]
  * *On using very large target vocabulary for neural machine translation*, arXiv:1412.2007 / ACL 2015 [[Paper](http://arxiv.org/pdf/1412.2007.pdf)]
* Univ. Montreal + Middle East Tech. Univ. + Univ. Maine [[Paper](http://arxiv.org/pdf/1503.03535.pdf)]
  * *On Using Monolingual Corpora in Neural Machine Translation*, arXiv:1503.03535
* Google [[Paper](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)]
  * *Sequence to Sequence Learning with Neural Networks*, arXiv:1409.3215 / NIPS 2014
* Google + NYU [[Paper](http://arxiv.org/pdf/1410.8206)]
  * *Addressing the Rare Word Problem in Neural Machine Transltaion*, arXiv:1410.8206 / ACL 2015
* ICT + Huawei [[Paper](http://arxiv.org/pdf/1506.06442.pdf)]
  * *A Deep Memory-based Architecture for Sequence-to-Sequence Learning*, arXiv:1506.06442
* Stanford [[Paper](http://arxiv.org/pdf/1508.04025.pdf)]
  * *Effective Approaches to Attention-based Neural Machine Translation*, arXiv:1508.04025
* Middle East Tech. Univ. + NYU + Univ. Montreal [[Paper](http://arxiv.org/pdf/1601.01073.pdf)]
  * *Multi-Way, Multilingual Neural Machine Translation with a Shared Attention Mechanism*, arXiv:1601.01073

#### Conversation Modeling
* *Neural Responding Machine for Short-Text Conversation*, arXiv:1503.02364 / ACL 2015 [[Paper](http://arxiv.org/pdf/1503.02364)]
* *A Neural Conversational Model*, arXiv:1506.05869 [[Paper](http://arxiv.org/pdf/1506.05869)]
* *The Ubuntu Dialogue Corpus: A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems*, arXiv:1506.08909 [[Paper](http://arxiv.org/pdf/1506.08909)]
* *Evaluating Prerequisite Qualities for Learning End-to-End Dialog Systems*, arXiv:1511.06931 [[Paper](http://arxiv.org/pdf/1511.06931)]
* *Dialog-based Language Learning*, arXiv:1604.06045, [[Paper](http://arxiv.org/pdf/1604.06045)]
* *Learning End-to-End Goal-Oriented Dialog*, arXiv:1605.07683 [[Paper](http://arxiv.org/pdf/1605.07683)]

#### Question Answering
* FAIR
  * *Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks*, arXiv:1502.05698 [[Web](https://research.facebook.com/researchers/1543934539189348)] [[Paper](http://arxiv.org/pdf/1502.05698.pdf)]
  * *Simple Question answering with Memory Networks*, arXiv:1506.02075 [[Paper](http://arxiv.org/abs/1506.02075)]
  * "The Goldilocks Principle: Reading Children's Books with Explicit Memory Representations", ICLR 2016 [[Paper](http://arxiv.org/abs/1511.02301)]
* DeepMind + Oxford [[Paper](http://arxiv.org/pdf/1506.03340.pdf)]
  * *Teaching Machines to Read and Comprehend*, arXiv:1506.03340 / NIPS 2015
* MetaMind [[Paper](http://arxiv.org/pdf/1506.07285.pdf)]
  * *Ask Me Anything: Dynamic Memory Networks for Natural Language Processing*, arXiv:1506.07285

### Computer Vision

#### Object Recognition
* *Recurrent Convolutional Neural Networks for Scene Labeling*, ICML 2014 [[Paper](http://jmlr.org/proceedings/papers/v32/pinheiro14.pdf)]
* *Recurrent Convolutional Neural Network for Object Recognition*, CVPR 2015 [[Paper](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Liang_Recurrent_Convolutional_Neural_2015_CVPR_paper.pdf)]
* *Scene Labeling with LSTM Recurrent Neural Networks*, CVPR 2015 [[Paper](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Byeon_Scene_Labeling_With_2015_CVPR_paper.pdf)]
* *Recurrent Convolutional Neural Networks for Object-Class Segmentation of RGB-D Video*, IJCNN 2015 [[Paper](http://www.ais.uni-bonn.de/papers/IJCNN_2015_Pavel.pdf)]
*  *Conditional Random Fields as Recurrent Neural Networks*, arXiv:1502.03240 [[Paper](http://arxiv.org/pdf/1502.03240)]
* *Semantic Object Parsing with Local-Global Long Short-Term Memory*, arXiv:1511.04510 [[Paper](http://arxiv.org/pdf/1511.04510.pdf)]
* *Inside-Outside Net: Detecting Objects in Context with Skip Pooling and Recurrent Neural Networks*, arXiv:1512.04143 / ICCV 2015 workshop [[Paper](http://arxiv.org/pdf/1512.04143)]

#### Visual Tracking
* *First Step toward Model-Free, Anonymous Object Tracking with Recurrent Neural Networks*, arXiv:1511.06425 [[Paper](http://arxiv.org/pdf/1511.06425)]


#### Image Generation
* *DRAW: A Recurrent Neural Network for Image Generation,* ICML 2015 [[Paper](http://arxiv.org/pdf/1502.04623)]
* *Unveiling the Dreams of Word Embeddings: Towards Language-Driven Image Generation,* arXiv:1506.03500 [[Paper](http://arxiv.org/pdf/1506.03500)]
* *Generative Image Modeling Using Spatial LSTMs,* arXiv:1506.03478 / NIPS 2015 [[Paper](http://arxiv.org/pdf/1506.03478)]
* *Pixel Recurrent Neural Networks,* arXiv:1601.06759 [[Paper](http://arxiv.org/abs/1601.06759)]

#### Video Analysis

* Univ. Toronto [[paper](http://arxiv.org/abs/1502.04681)]
  * *Unsupervised Learning of Video Representations using LSTMs*, arXiv:1502.04681 / ICML 2015
* Univ. Cambridge [[paper](http://arxiv.org/abs/1511.06309)]
  * *Spatio-temporal video autoencoder with differentiable memory*, arXiv:1511.06309





#### Turing Machines
* *Neural Turing Machines,* arXiv preprint arXiv:1410.5401 [[Paper](http://arxiv.org/pdf/1410.5401)]
* *Memory Networks,* arXiv:1410.3916 [[Paper](http://arxiv.org/pdf/1410.3916)]
* *Inferring Algorithmic Patterns with Stack-Augmented Recurrent Nets*, arXiv:1503.01007 / NIPS 2015 [[Paper](http://arxiv.org/pdf/1503.01007)]
* *End-To-End Memory Networks*, arXiv:1503.08895 / NIPS 2015 [[Paper](http://arxiv.org/pdf/1503.08895)]
* *Reinforcement Learning Neural Turing Machines,* arXiv:1505.00521 [[Paper](http://arxiv.org/pdf/1505.00521)]
* *Recurrent Neural Networks with External Memory for Language Understanding*, arXiv:1506.00195 [[Paper](http://arxiv.org/pdf/1506.00195.pdf)]
* *A Deep Memory-based Architecture for Sequence-to-Sequence Learning*, arXiv:1506.06442 [[Paper](http://arxiv.org/pdf/1506.06442.pdf)]
* A*Neural Programmer: Inducing Latent Programs with Gradient Descent*, arXiv:1511.04834 [[Paper](http://arxiv.org/pdf/1511.04834.pdf)]
* *Neural Programmer-Interpreters*, arXiv:1511.06279 [[Paper](http://arxiv.org/pdf/1511.06279.pdf)]
* *Neural Random-Access Machines*, arXiv:1511.06392 [[Paper](http://arxiv.org/pdf/1511.06392.pdf)]
* Łukasz Kaiser and Ilya Sutskever, *Neural GPUs Learn Algorithms*, arXiv:1511.08228 [[Paper](http://arxiv.org/pdf/1511.08228.pdf)]
* *Skip-Thought Memory Networks*, arXiv:1511.6420 [[Paper](https://pdfs.semanticscholar.org/6b9f/0d695df0ce01d005eb5aa69386cb5fbac62a.pdf)]
* *Learning Simple Algorithms from Examples*, arXiv:1511.07275 [[Paper](http://arxiv.org/pdf/1511.07275.pdf)]

### Robotics

* *Listen, Attend, and Walk: Neural Mapping of Navigational Instructions to Action Sequences*, arXiv:1506.04089 [[Paper](http://arxiv.org/pdf/1506.04089.pdf)]
* *Policy Learning with Continuous Memory States for Partially Observed Robotic Control,* arXiv:1507.01273. [[Paper]](http://arxiv.org/pdf/1507.01273)

### Other
* *Generating Sequences With Recurrent Neural Networks,* arXiv:1308.0850 [[Paper]](http://arxiv.org/abs/1308.0850)
* *Recurrent Models of Visual Attention*, NIPS 2014 / arXiv:1406.6247 [[Paper](http://arxiv.org/pdf/1406.6247.pdf)]
* *Learning to Execute*, arXiv:1410.4615 [[Paper](http://arxiv.org/pdf/1410.4615.pdf)] [[Code](https://github.com/wojciechz/learning_to_execute)]
* *Scheduled Sampling for Sequence Prediction with
Recurrent Neural Networks*, arXiv:1506.03099 / NIPS 2015 [[Paper](http://arxiv.org/pdf/1506.03099)]
* *DAG-Recurrent Neural Networks For Scene Labeling*, arXiv:1509.00552 [[Paper](http://arxiv.org/pdf/1509.00552)]
* *Recurrent Spatial Transformer Networks*, arXiv:1509.05329 [[Paper](http://arxiv.org/pdf/1509.05329)]
* *Batch Normalized Recurrent Neural Networks*, arXiv:1510.01378 [[Paper](http://arxiv.org/pdf/1510.01378)]
* *Deeply-Recursive Convolutional Network for Image Super-Resolution*, arXiv:1511.04491 [[Paper]](http://arxiv.org/abs/1511.04491)
* *First Step toward Model-Free, Anonymous Object Tracking with Recurrent Neural Networks*, arXiv:1511.06425 [[Paper](http://arxiv.org/pdf/1511.06425.pdf)]
* *ReSeg: A Recurrent Neural Network for Object Segmentation*, arXiv:1511.07053 [[Paper](http://arxiv.org/pdf/1511.07053.pdf)]
* *On Learning to Think: Algorithmic Information Theory for Novel Combinations of Reinforcement Learning Controllers and Recurrent Neural World Models*, arXiv:1511.09249 [[Paper]](http://arxiv.org/pdf/1511.09249)


------------------------


----------------------
