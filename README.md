Created an Automated system that predicts the judgement of a case file and gives its corresponding explanation for that decision. 

I have used a hierarchical model for the prediction task, where the first level is XLNET and the second level is BiGRu with att, the output of the first layer is an CLS Embedding which is passed to the BiGRU layer which gives the probability distribution which is sent to a softmax function and the final prediction label is generated.

The Model has achieved an accuracy of 74% which is higher than most of the previous work, hence making the project exempelary
