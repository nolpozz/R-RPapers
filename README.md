# R-RPapers
Record of NLP and Linguistics Papers I've Read


In Winter Quarter of 2025(January-March) I took on a Research Assistant role with Dr. Zhewei Sun of TTIC. Since I applied for credit(CMSC29700 Reading&Research) for the position, it was an unpaid position.

Over winter break, Dr. Sun asked me to read through a number of white papers so that I was versed in the history of NLP practices leading up to the project. Here is a record of those papers. Since I wanted to read these papers thoroughly and understand the concepts deeply, it cut in greatly to the time I put into my projects. Because of this, I wanted to have public record of this time. 

I am partly doing a side project as well, which I will upload once/if it is completed in which I am implementing a Seq2Seq model for English and Russian on google colab based on this tensorflow tutorial https://www.tensorflow.org/text/tutorials/nmt_with_attention. 

For those interested, the subject matter for the quarter is as follows: The proposed research project stdies statistical patterns in multilingual slang sense extensions and potential applications in natural language processing. The first phase of the project investigates whether sufficient regularities exist across attested slang usages from different languages/cultures to allow transfer learning of slang semantics from standard American English or other English variants or languages. Contingent on this finding, the research will recommend/develop efficient NLP methods to process slang used in other languages/cultures. 

At the end of the quarter, if a paper is published, I will link to it here.


With that being said
Here are the papers I have read: 
Word Embeddings:
Word2Vec:

Efficient Estimation of Word Representations in
 Vector Space, Mikolov, https://arxiv.org/pdf/1301.3781.pdf
Distributed Representations of Words and Phrases
 and their Compositionality, Mikolov, https://proceedings.neurips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf
 
GloVe:
 GloVe: Global Vectors for Word Representation, Pennington, https://nlp.stanford.edu/pubs/glove.pdf
 
Learning fine-grained vectors:
Distributed Representations of Geographically Situated Language, Bamman, https://aclanthology.org/P14-2134.pdf
 
Byte-pair encoding (BPE):
 Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016. Neural Machine Translation of Rare Words with Subword Units. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1715–1725, Berlin, Germany. Association for Computational Linguistics. https://aclanthology.org/P16-1162/
 
fastText:
 Enriching Word Vectors with Subword Information (Bojanowski et al., TACL 2017) https://aclanthology.org/Q17-1010/
 
SentencePiece:
 SentencePiece: A simple and language independent subword tokenizer
 and detokenizer for Neural Text Processing, Kudo, https://aclanthology.org/D18-2012.pdf
 
 
Attention/Transformers/LLMs:
 
Seq2Seq:
 Sequence to Sequence Learning with Neural Networks, Sutskever, https://arxiv.org/abs/1409.3215
 
Seq2Seq w/ attention:
 Neural Machine Translation by Jointly Learning to Align and Translate, Bahdanau, https://arxiv.org/abs/1409.0473
 
Transformers:
 Attention Is All You Need, Vaswani, https://arxiv.org/abs/1706.03762
 
ELMo:
 Deep Contextualized Word Representations, Peters, https://aclanthology.org/N18-1202/
 
BERT:
 BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, Devlin, https://aclanthology.org/N19-1423/
 
GPT:
 Improving Language Understanding by Generative Pre-Training, Radford, https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
 Language Models are Unsupervised Multitask Learners, Radford, https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
 
Sparse transformers:
 Generating Long Sequences with Sparse Transformers, Child, https://arxiv.org/abs/1904.10509
 
RLHF:
 Training language models to follow instructions with human feedback, Ouyang, https://arxiv.org/abs/2203.02155
 
Attention cache:
 Efficient Streaming Language Models with Attention Sinks, Xiao, https://arxiv.org/abs/2309.17453
SirLLM: Streaming Infinite Retentive LLM, Yao, https://aclanthology.org/2024.acl-long.143/
 
Efficient Fine-tuning (LoRA):
 LoRA: Low-Rank Adaptation of Large Language Models, Hu, https://arxiv.org/abs/2106.09685
QLoRA: Efficient Finetuning of Quantized LLMs, Dettmers, https://arxiv.org/abs/2305.14314
 
 
Potentially relevant techniques:
 
Sentence-BERT
 Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks, Reimers, https://aclanthology.org/D19-1410/
 
BERTScore:
 BERTScore: Evaluating Text Generation with BERT, Zhang, https://iclr.cc/virtual_2020/poster_SkeHuCVFDr.html
 
Edge probing:
 What do you learn from context? Probing for sentence structure in contextualized word representations, Tenney, https://arxiv.org/abs/1905.06316
 
SHAP values:
A Unified Approach to Interpreting Model Predictions, Lundberg, https://arxiv.org/pdf/1705.07874
 
CCA:
 Insights on representational similarity in neural networks with canonical correlation, Morcos, https://arxiv.org/pdf/1806.05759
 
 
NLP for slang:
 
Dr. Sun's Thesis. Chapter 1 (What is slang), 3 (Generation) will be especially relevant for this project:
 
https://zhewei-sun.github.io/files/nlp_for_slang_thesis.pdf

 
Dr. Sun's recent work with a new slang dataset based on OpenSubtitles:
 
https://aclanthology.org/2024.naacl-long.94/

 
Dr. Sun's work on generation focused on the case of reuse (i.e. recycling an existing word form) and this paper by Kulkarni and Wang looks at generating novel forms:
 
https://aclanthology.org/N18-1129/
 
other papers on slang/jargon:
 Slangvolution: A Causal Analysis of Semantic Change and Frequency Dynamics in Slang, Keidar, https://aclanthology.org/2022.acl-long.101/

Characterizing English Variation across Social Media Communities with BERT, Lucy, https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00383/101877/Characterizing-English-Variation-across-Social 



 As I read more, I will try to include them here as well in a seperate section. 
