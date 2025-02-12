import streamlit as st
import nltk
import numpy as np
from transformers import *
from keras.models import Model
from keras import layers
from keras.preprocessing import sequence
from nltk.tokenize import word_tokenize
from keras.models import load_model
import itertools
from transformers import XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig
from keras.layers import Embedding, Dense, Input, concatenate, Layer, Lambda, Dropout, Activation
import keras
import os
import random
import pandas as pd
import numpy as np
import csv
import tensorflow as tf
import torch
from sklearn.model_selection import train_test_split
import textwrap
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import time
import datetime
import json
import nltk
from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from transformers import XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig
from transformers import XLMForSequenceClassification, XLMTokenizer, XLMConfig
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig
from transformers import DebertaForSequenceClassification, DebertaTokenizer, DebertaConfig
import os
from PyPDF2 import PdfReader
import shutil
import re
from keras import Sequential
from keras.utils import Sequence
from keras.layers import LSTM, Dense, Masking, GRU
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Embedding, Dense, Input, concatenate, Layer, Lambda, Dropout, Activation
import datetime
from datetime import datetime
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, TensorBoard
import tensorflow as tf
from keras import initializers as initializers, regularizers, constraints
from keras import backend as K
import numpy as np
from numpy import load
import pandas as pd
from keras import layers
from tensorflow import keras
import itertools
from transformers import AutoTokenizer, AutoModel

import os
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import numpy as np
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from PIL import Image

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_together import Together
import os
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
import time

from Comb_FINAL import extract_text_and_headnote_from_pdf,generate_np_files_for_emb,generate_summary,get_explanation,new_test_generator,AttentionLayer,xlnet_tokenize,sentence_marker,chunked_tokens_maker,chatbot_ui
nltk.download('punkt_tab')
nltk_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

Summary_tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
Summary_model = AutoModel.from_pretrained("law-ai/InLegalBERT")


modelDirectory="XLNet_FINAL"
tokenizer = XLNetTokenizer.from_pretrained(modelDirectory)
model = XLNetForSequenceClassification.from_pretrained(modelDirectory, output_hidden_states=True)
device="cuda:0"
model.to(device)

text_input = Input(shape=(None,3072,), dtype='float32', name='text') #1 sent_input
#print("text input",text_input)
l_mask = layers.Masking(mask_value=-99.)(text_input) #2 sent_encoder
#print("l_mask",l_mask)
# Which we encoded in a single vector via a LSTM
encoded_text = layers.Bidirectional(GRU(200,return_sequences=True))(l_mask) #3 sent_gru
encoded_text1 = layers.Bidirectional(GRU(200,return_sequences=True))(encoded_text) #4
encoded_text2 = layers.Bidirectional(GRU(200,return_sequences=True))(encoded_text1)
# out_dense = layers.Dense(200, activation='relu')(encoded_text1) #5 sent_dense
sent_att,sent_coeffs, = AttentionLayer(400,return_coefficients=True,name='sent_attention')(encoded_text2) #6
# print("sent_att",sent_att)
# print("sent_coeffs",sent_coeffs)
sent_drop = Dropout(0.5,name='sent_dropout')(sent_att)
# And we add a softmax classifier on top
out1 = layers.Dense(64, activation='relu')(sent_drop) #7 preds
out = layers.Dense(1, activation='sigmoid')(out1)
modelPred = Model(text_input,out)
modelPred.load_weights('Prediction_full_xlnet/XGA_concat_epoch1_3.h5')
modelPred.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
level1PathOutput="L1_Output/embeds.npy"
exp_pred=0 ## GLOBAL VARIABLE

pdf_path=""

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #f5f5f5;
        color: #333;
    }
    .main-header {
        font-size: 2.5rem;
        color: #4CAF50;
        font-weight: bold;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #FF5722;
        margin-top: 1rem;
        text-align: center;
    }
    .uploaded-pdf {
        margin-top: 1.5rem;
        font-size: 1.1rem;
        color: #3F51B5;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 1rem;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)
def main():
    st.markdown('<div class="main-header">Legal Case Prediction, Summary,Explanation & LegalSmart Chatbot</div>', unsafe_allow_html=True)
    
    st.sidebar.image("https://img.icons8.com/ios-filled/100/4CAF50/law.png", width=100)

    # Navigation Sidebar
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Summarization", "Prediction", "Explanation", "LegalSmart Chatbot"])

    # Space between sections
    add_vertical_space(2)

    if "pdf_path" not in st.session_state:
        st.session_state.pdf_path = None

        

    if selection == "Summarization":
        colored_header("Case Summary", color_name="green-70")
        global pdf_path
        uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
        
        if uploaded_file:
            st.session_state.pdf_path = uploaded_file
            extracted_text, headnote = extract_text_and_headnote_from_pdf(st.session_state.pdf_path)
            case_summary = generate_summary(Summary_model, Summary_tokenizer, extracted_text)
            st.markdown('<div class="uploaded-pdf">Processing your PDF...</div>', unsafe_allow_html=True)
            # Example of displaying summary
            st.write("Your case summary will appear here.")
            st.write(case_summary)
            
            
    elif selection == "Prediction":
        colored_header("Case Prediction", color_name="orange-70")
        
        
        ###pdf_path = st.file_uploader("Upload a PDF", type="pdf")
        print("PDF PATH",st.session_state.pdf_path)
        if st.session_state.pdf_path:
            st.markdown('<div class="uploaded-pdf">Analyzing your case...</div>', unsafe_allow_html=True)
            extracted_text, headnote = extract_text_and_headnote_from_pdf(st.session_state.pdf_path)
            # Generate embeddings
            embeds = generate_np_files_for_emb(model, extracted_text, tokenizer)
            np.save(level1PathOutput, embeds)
            x_test0 = load(level1PathOutput, allow_pickle= True)
            new_test_x = []
            for i in range(len(x_test0)):
              full_emb = x_test0[i]
              new_test_x.append(full_emb)
              for j in range(len(full_emb)):
                emb_occluded = []
                for k in range(len(full_emb)):
                  if(k==j):
                    emb_occluded.append(np.zeros(768))
                  else:
                    emb_occluded.append(full_emb[k])
            
                new_test_x.append(emb_occluded)
            
            num_features = 3072
            num_sequences_test = len(new_test_x)
            test_gen = new_test_generator(num_features,new_test_x)
            
            # print("\nStarting evaluation...")
            # model.evaluate(new_test_generator(), steps=batches_per_epoch_test)
            preds = modelPred.predict(test_gen)
            
            scores = []
            
            startpt = 0
            for i in range(len(x_test0)):
              n_emb_for_this_doc = x_test0[i].shape[0]
              tot_embs = n_emb_for_this_doc + 1
              act_pred = preds[startpt] > 0.5
              startpt += tot_embs
            
            
            ## FINAL Prediction
            #print(act_pred[0])
            #print("="*100)
            ##st.write("Embeddings generated.")
            # Perform prediction
            global exp_pred
            exp_pred= 1 if act_pred[0] else 0
            prediction = "Accepted" if act_pred[0] else "Rejected"
            st.markdown(f"### **The prediction for this case is: {prediction}**")

    
    elif selection == "Explanation":
        colored_header("Case Explanation", color_name="blue-70")
        # pdf_path = st.file_uploader("Upload a PDF", type="pdf")
        # if pdf_path:
        if st.session_state.pdf_path:
            st.markdown('<div class="uploaded-pdf">Generating explanation...</div>', unsafe_allow_html=True)
            extracted_text, headnote = extract_text_and_headnote_from_pdf(st.session_state.pdf_path)
            chunk_scores = load("xlnet_embeds/xlnet_occwts.npy", allow_pickle = True)
            chunk_scores = list(chunk_scores)
            sents = nltk_tokenizer.tokenize(extracted_text)
            xlnet_tokenized_sents = xlnet_tokenize(sents, tokenizer)
            marked_tokenized_sents = sentence_marker(xlnet_tokenized_sents)
            xlnet_tokens = list(itertools.chain.from_iterable(xlnet_tokenized_sents))
            markers = list(itertools.chain.from_iterable(marked_tokenized_sents))
            
            if len(xlnet_tokens) > 10000:
                xlnet_tokens = xlnet_tokens[len(xlnet_tokens) - 10000:]
                markers = markers[len(markers) - 10000:]
            
            chunked_xlnet_tokens, chunked_markers = chunked_tokens_maker(xlnet_tokens, markers)
            explanation_of_this_doc = get_explanation(model,chunked_xlnet_tokens, chunked_markers, chunk_scores,1, tokenizer, exp_pred)

            st.write(f"### **Your case explanation will appear here.**")
            st.write(explanation_of_this_doc)
            
    elif selection == "LegalSmart Chatbot":
        chatbot_ui()
        
if __name__ == "__main__":
    main()

