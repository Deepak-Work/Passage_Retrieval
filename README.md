# Passage_Retrieval
Solution of Microsoft's AI Challenge 2018 involving passage retrieval from a given dataset using Machine Learning Models.
The competition is now open for learning purposes only.<br>
Link: https://competitions.codalab.org/competitions/20616

<H2> Implementation </H2>

<B>BM25</B><br>
The famous probabilistic method of passage retrieval is implemented with a few tweaks here and there.
Reference: http://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf<br>

<B>DL</B><br>
There are quite a few Deep Learning Models present for passage retrieval problems with CNN, using basic LSTM, attention with LSTM being a few of the many. Here we have implemented a BIDAF model. 
Reference: https://arxiv.org/abs/1611.01603

<H2>Usage</H2>

Due to limited resources, the above model was trained on an 8 GB GeForce 960M GPU and 8 GB RAM. Better GPU is prefferable or it could take a lot of time.<br>
Being lazy as I am, I'm not attaching a requirements.txt file. Very basic tensorflow libraries have been used. Please go through the code for exact details.<br>
The data is expected to be downloaded from the competion's link mentioned above and glove word embeddings requires some googling [pretty simple one though]. Here I have used the 100 dim embeddings, you can make the appropriate changes as you please. <br>
<br>
<B>To reach the output, run the codes in the following order:<br>
lstm1D_Attention.py<br>
BM25.py<br>
Ensemble.py<br>
</B>

<H2>Better Ideas for Future work</H2>

Due to lack of hardware resources, all the models thought of couldn't be implemented.<br>
Using pre-trained <B>elmo embeddings</B> and fine tuning with the data should give better results<br>


