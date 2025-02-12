import os
os.environ["KERAS_BACKEND"] = "tensorflow"

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



import torch
import numpy as np

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_together import Together
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
import time

def chatbot_ui():
    # Set up page configuration and styling
    

    # Styling
    st.markdown(
        """
        <style>
        /* Global styling for colors, fonts, and layout */
        body {
            background-color: #f5f7fa;
            color: #333333;
            font-family: Arial, sans-serif;
        }
        /* Header styling for chatbot title */
        .header {
            text-align: center;
            font-size: 3em;
            color: #0d47a1;
            font-weight: bold;
            margin-top: 20px;
        }
        /* Full-width image styling */
        .full-width-image {
            margin: 0 auto;
            width: 100%;
            height: auto;
            display: block;
        }
        /* Custom button styling */
        div.stButton > button:first-child {
            background-color: #0d47a1;
            color: white;
            border-radius: 8px;
            font-size: 16px;
            padding: 10px 20px;
        }
        div.stButton > button:active {
            background-color: #0b3954;
        }
        /* Chat window and input field styling */
        .stChatMessageUser {
            background-color: #bbdefb;
            border-radius: 10px;
            padding: 10px;
            margin: 5px 0;
        }
        .stChatMessageAssistant {
            background-color: #e1f5fe;
            color: #333333;
            border-radius: 10px;
            padding: 10px;
            margin: 5px 0;
        }
        /* Hide default Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDecoration {display:none;}
        button[title="View fullscreen"] {
            visibility: hidden;
        }
        </style>
        """,
    unsafe_allow_html=True,
    )

 

    # Full-width image display
    image_path = "chatbot_legal.jpeg"  # Ensure this path points to your image
    try:
        st.image(image_path, use_container_width=True, output_format="JPEG")
    except FileNotFoundError:
        st.error("Image not found. Please check the path to 'chatbot_legal.jpeg' and try again.")

    # Define reset conversation function
    def reset_conversation():
        st.session_state.messages = []
        st.session_state.memory.clear()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)

    embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1", model_kwargs={"trust_remote_code":True, "revision":"289f532e14dbbbd5a04753fa58739e9ba766f3c7"})
    db = FAISS.load_local("law_vector_db", embeddings, allow_dangerous_deserialization=True)
    db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    prompt_template = prompt_template ="""
<s>[INST] You are a highly knowledgeable and professional legal chatbot trained on the Indian Penal Code (IPC), Indian Constitution, and Bharatiya Nyaya Sanhita (BNS). Your role is to assist users by answering legal questions accurately, concisely, contextually, and professionally.

### Guidelines:
1. **Dual Audience Handling**:
   - **For naive users**:
     - Use simple, clear, and conversational language.
     - Avoid legal jargon unless necessary, and explain terms clearly when used.
     - Provide relevant examples or analogies to simplify complex legal concepts.
     - Be concise but ensure users understand the key points.
   - **For legal professionals**:
     - Use precise legal terminology and maintain a professional tone.
     - Cite relevant sections, articles, or clauses explicitly.
     - Provide detailed and authoritative insights while avoiding unnecessary simplification.
     - Cross-reference relevant laws and precedents when applicable.

2. **Accuracy and Context**:
   - Always specify the source of your response, such as IPC sections, Constitution articles, or BNS chapters.
   - Tailor responses to the user's context based on prior chat history and the provided query.
   - If a question requires further clarification, politely ask for more details.

3. **Limitations**:
   - Do not provide speculative advice, personal opinions, or non-legal interpretations.
   - If a query is beyond your training, suggest consulting a qualified legal professional.
   - Avoid offering solutions to cases outside the scope of Indian law.

### Format:
- **CONTEXT**: Relevant information or previous details from the user.
- **CHAT HISTORY**: A summary of the previous conversation (if applicable).
- **QUESTION**: The user's current question or query.
- **ANSWER**: Your response to the user's question,

### Example Interactions:

#### Example 1: Naive User
**CONTEXT**: The user is asking about their constitutional rights.
**CHAT HISTORY**: None
**QUESTION**: What does the right to freedom of speech mean?
**ANSWER**: Since you are a naive user asking about the right to freedom of speech, the right to freedom of speech is guaranteed under Article 19(1)(a) of the Indian Constitution. It allows individuals to express their opinions freely, as long as it doesn't harm public order, morality, or the sovereignty of India.

#### Example 2: Legal Professional
**CONTEXT**: The user is a legal practitioner seeking details on IPC Section 302.
**CHAT HISTORY**: None
**QUESTION**: Can you explain the provisions under IPC Section 302?
**ANSWER**: Since you are a legal professional asking about IPC Section 302, IPC Section 302 prescribes the punishment for murder. It states that whoever commits murder shall be punished with death or life imprisonment and may also be liable to pay a fine. The definition of "murder" is detailed in Section 300 of the IPC, which outlines the conditions under which an act constitutes murder.

#### Example 3: Ambiguous Query
**CONTEXT**: The user has not provided sufficient details.
**CHAT HISTORY**: None
**QUESTION**: What can I do if my neighbor violates the law?
**ANSWER**: Could you clarify the specific issue with your neighbor? For example, are you referring to noise complaints, property disputes, or another concern? This will help me provide a more relevant response.

#### Example 4: Cross-Referencing (BNS and IPC)
**CONTEXT**: The user is interested in understanding the updates in Bharatiya Nyaya Sanhita.
**CHAT HISTORY**: None
**QUESTION**: How does the BNS address sedition compared to IPC?
**ANSWER**: Since you are asking about how the BNS addresses sedition compared to IPC, IPC Section 124A deals with sedition and prescribes punishment for actions or speech inciting hatred or rebellion against the government. The Bharatiya Nyaya Sanhita (BNS) revises this section, focusing on penalizing specific acts of subversion while excluding criticism of the government to uphold free speech. Let me know if you'd like detailed comparisons.

### Input Fields:
- **CONTEXT**: {context}
- **CHAT HISTORY**: {chat_history}
- **QUESTION**: {question}
- **ANSWER**: </s>[INST]
"""




    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question', 'chat_history'])

    TOGETHER_AI_API= 'd5b19df5283427f0301b6a6e23d463cebcbe8ba62ca2079a61a14d1897d147ae'
    llm = Together(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        temperature=0.5,
        max_tokens=1024,
        together_api_key=f"{TOGETHER_AI_API}"
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=st.session_state.memory,
        retriever=db_retriever,
        combine_docs_chain_kwargs={'prompt': prompt}
    )

    # Display chat history and user inputs
    for message in st.session_state.messages:
        if message.get("role") == "user":
            st.chat_message("user").write(message.get("content"))
        else:
            st.chat_message("assistant").write(message.get("content"))

    # Input field for new user messages
    input_prompt = st.chat_input("Ask your legal question here...")

    if input_prompt:
        st.session_state.messages.append({"role": "user", "content": input_prompt})
    
        with st.chat_message("assistant"):
            with st.spinner("Thinking 💡..."):
                result = qa.invoke(input=input_prompt)
                message_placeholder = st.empty()
                full_response = "⚠️ **_Note: Information provided may be inaccurate._** \n\n"
            
                for chunk in result["answer"]:
                    full_response += chunk
                    time.sleep(0.02)
                    message_placeholder.markdown(full_response + " ▌")
                
            st.button('Reset All Chat 🗑️', on_click=reset_conversation)

        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})




def generate_summary(model, tokenizer, text):
    print("CAME INSIDE generate_summary function")
    
    # Check if CUDA is available and set the device accordingly
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    tokenizer.max_len = model.config.max_position_embeddings

    # Encode the entire text and move to GPU
    encoded_input = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    output = model(**encoded_input)
    embeddings = output.pooler_output

    # Step 1: Compute sentence embeddings
    sentences = text.split('. ')  # Split text into sentences

    # Calculate the desired number of sentences for the summary based on the percentage
    percentage = 0.08  # Adjust the desired percentage as needed
    num_sentences = int(len(sentences) * percentage)

    # Adjust if the number of sentences is less than the calculated value
    if len(sentences) < num_sentences:
        num_sentences = len(sentences)

    sentence_embeddings = []
    for sentence in sentences:
        encoded_input = tokenizer(sentence, return_tensors="pt", truncation=True).to(device)
        output = model(**encoded_input)
        sentence_embedding = output.pooler_output.squeeze(dim=0)
        sentence_embeddings.append(sentence_embedding.detach().cpu().numpy())  # Move to CPU for NumPy

    # Step 2: Compute similarity scores
    similarity_scores = []
    for i, embedding_i in enumerate(sentence_embeddings):
        similarity_scores_i = []
        for j, embedding_j in enumerate(sentence_embeddings):
            similarity = np.dot(embedding_i, embedding_j) / (
                np.linalg.norm(embedding_i) * np.linalg.norm(embedding_j)
            )
            similarity_scores_i.append(similarity)
        similarity_scores.append(similarity_scores_i)

    similarity_scores = np.array(similarity_scores)

    # Step 3: Select top sentences
    summary_indices = np.argsort(-similarity_scores.sum(axis=1))[:num_sentences]
    summary_indices.sort()
    summary_sentences = [sentences[i] for i in summary_indices]
    InLegal_summary = '. '.join(summary_sentences)
    
    return InLegal_summary


# Path to the judgment and summary folders

def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

import tensorflow as tf
from tensorflow.keras import layers, initializers

class AttentionLayer(layers.Layer):
    """
    Hierarchical Attention Layer as described by Hierarchical Attention Networks for Document Classification (2016)
    - Yang et. al.
    Source: https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf
    TensorFlow backend
    """
    def __init__(self, attention_dim=200, return_coefficients=False, **kwargs):
        # Initializer
        self.supports_masking = True
        self.return_coefficients = return_coefficients
        self.init = initializers.get('glorot_uniform')  # initializes values with uniform distribution
        self.attention_dim = attention_dim
        print("attention_dim", self.attention_dim)
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Builds all weights
        # W = Weight matrix, b = bias vector, u = context vector
        assert len(input_shape) == 3
        # Use tf.Variable to define the trainable variables
        self.W = self.add_weight(name='W', shape=(input_shape[-1], self.attention_dim),
                                 initializer=self.init, trainable=True)
        self.b = self.add_weight(name='b', shape=(self.attention_dim,), initializer='zeros', trainable=True)
        self.u = self.add_weight(name='u', shape=(self.attention_dim, 1),
                                 initializer=self.init, trainable=True)

        super(AttentionLayer, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def _get_attention_weights(self, X):
        u_tw = tf.tanh(tf.matmul(X, self.W))  # Matrix multiplication with tf.matmul
        tw_stimulus = tf.matmul(u_tw, self.u)  # Matrix multiplication with tf.matmul

        # Remove the last axis and apply softmax to the stimulus to get a probability
        tw_stimulus = tf.squeeze(tw_stimulus, -1)
        tw_stimulus = tf.exp(tw_stimulus)

        tw_stimulus /= tf.cast(tf.reduce_sum(tw_stimulus, axis=1, keepdims=True) + tf.keras.backend.epsilon(),
                               tf.float32)  # Normalize and avoid division by zero

        att_weights = tf.expand_dims(tw_stimulus, -1)
        return att_weights
    def call(self, hit, mask=None):
      att_weights = self._get_attention_weights(hit)

      # Compute the attention-weighted input
      uit = tf.matmul(hit, self.W)  # Matrix multiplication with tf.matmul
      uit = tf.add(uit, self.b)  # Add bias term using tf.add
      uit = tf.tanh(uit)  # Apply tanh activation

      ait = tf.matmul(uit, self.u)  # Matrix multiplication with tf.matmul
      ait = tf.squeeze(ait, -1)  # Remove the last axis (dimension)
      ait = tf.exp(ait)  # Apply exponentiation to the result

      if mask is not None:
          ait *= tf.cast(mask, tf.float32)  # Apply the mask

      # Normalize attention weights
      ait /= tf.cast(tf.reduce_sum(ait, axis=1, keepdims=True) + tf.keras.backend.epsilon(), tf.float32)
      ait = tf.expand_dims(ait, -1)  # Add an extra dimension for broadcasting

      weighted_input = hit * ait  # Weighted sum of inputs

      if self.return_coefficients:
          return [tf.reduce_sum(weighted_input, axis=1), ait]  # Return the weighted sum and attention coefficients
      else:
          return tf.reduce_sum(weighted_input, axis=1)  # Return only the weighted sum

    def compute_output_shape(self, input_shape):
        if self.return_coefficients:
            return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[-1], 1)]
        else:
            return (input_shape[0], input_shape[-1])

def extract_text_and_headnote_from_pdf(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()

    text=text.lower()

    act_search = re.search(r"act:(.*)", text, re.DOTALL | re.IGNORECASE)
    if act_search:
        act_text = act_search.group(1).strip()
    else:
        act_text = "Act section not found."

    # Extract headnote using regular expression pattern
    headnote_search = re.search(r"headnote:(.*?)judgment:", text, re.DOTALL | re.IGNORECASE)
    if headnote_search:
        # Extract the headnote and remove leading/trailing spaces
        headnote = headnote_search.group(1).strip()
    else:
        headnote = "Headnote not found."

    # Remove any text that includes "indian kanoon" followed by a link
    act_text = re.sub(r"indian kanoon - http://indiankanoon\.org/.*?\n", "", act_text, flags=re.IGNORECASE)
    headnote = re.sub(r"indian kanoon - http://indiankanoon\.org/.*?\n", "", headnote, flags=re.IGNORECASE)

    
    return act_text, headnote



def att_masking(input_ids):
  attention_masks = []
  for sent in input_ids:
    att_mask = [int(token_id > 0) for token_id in sent]
    attention_masks.append(att_mask)
  return attention_masks

def grouped_input_ids(tokenizer,all_toks):
  splitted_toks = []
  l=0
  r=510
  while(l<len(all_toks)):
    splitted_toks.append(all_toks[l:min(r,len(all_toks))])
    l+=410
    r+=410

  CLS = tokenizer.cls_token
  SEP = tokenizer.sep_token
  e_sents = []
  for l_t in splitted_toks:
    l_t = l_t + [SEP] + [CLS]
    encoded_sent = tokenizer.convert_tokens_to_ids(l_t)
    e_sents.append(encoded_sent)

  e_sents = pad_sequences(e_sents, maxlen=512, value=0, dtype="long", padding="pre")
  att_masks = att_masking(e_sents)
  return e_sents, att_masks

def get_output_for_one_vec(model,input_id, att_mask):
    # Convert inputs to tensors and add batch dimension
    device="cuda:0"
    input_ids = torch.tensor(input_id).unsqueeze(0).to(device)
    att_masks = torch.tensor(att_mask).unsqueeze(0).to(device)
    # print("l1 part")
    # print(model.input)
    ##model.eval()
    with torch.no_grad():
        # Get the model outputs
        outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=att_masks)

    # Extract the mems (hidden representations)
    
    mems = outputs.hidden_states  # mems is a tuple of tensors
    log=outputs.logits
    # Determine the total number of layers in mems
    total_layers = len(mems)

    # Dynamically concatenate the last few layers (e.g., last 4 layers)
    num_layers_to_concat = min(4, total_layers)
    vecs = [mems[-(i + 1)][0][-1] for i in range(num_layers_to_concat)]

    # Concatenate and convert the resulting tensor to a numpy array
    vec = torch.cat(vecs, dim=0).detach().cpu().numpy()
    return vec,log


def generate_np_files_for_emb(model,extracted_text, tokenizer):
    all_docs=[]
    toks = tokenizer.tokenize(extracted_text)
    if(len(toks) > 10000):
      toks = toks[len(toks)-10000:]

    splitted_input_ids, splitted_att_masks = grouped_input_ids(tokenizer,toks)

    vecs = []
    for index,ii in enumerate(splitted_input_ids):
        v,_=get_output_for_one_vec(model,ii, splitted_att_masks[index])
        vecs.append(v)

    one_doc = np.asarray(vecs)
    all_docs.append(one_doc)


    all_docs = np.asarray(all_docs,dtype = object)
    return all_docs



def preprocess_data(x_list):
    """
    Preprocess a list of sequences to ensure each has 8 frames with 3072 features.
    Handles variable-length sequences through padding or truncation.
    
    Args:
        x_list: List of sequences, where each sequence can be numpy array or list of frames
        
    Returns:
        List of preprocessed sequences, each shaped (8, 3072)
    """
    processed_x = []
    target_frames = 8
    print(f"Starting preprocessing of {len(x_list)} sequences...")
    
    for i, x in enumerate(x_list):
        if i % 1000 == 0:
            print(f"Processing item {i}/{len(x_list)}")
        try:
            if isinstance(x, np.ndarray):
                # Handle numpy arrays
                if x.shape[1] == 3072:  # Correct feature dimension
                    if x.shape[0] == target_frames:
                        processed_x.append(x)
                    else:
                        # Pad or truncate to 8 frames
                        if x.shape[0] < target_frames:
                            # Pad with zeros
                            padding = np.zeros((target_frames - x.shape[0], 3072), dtype=np.float32)
                            sequence = np.vstack([x, padding])
                        else:
                            # Truncate to first 8 frames
                            sequence = x[:target_frames]
                        processed_x.append(sequence)
                else:
                    print(f"Unexpected feature dimension at index {i}: {x.shape}")
                    processed_x.append(np.zeros((target_frames, 3072), dtype=np.float32))
            else:
                # Handle list of frames
                processed_frames = []
                for frame in x:
                    frame_array = np.array(frame, dtype=np.float32)
                    if frame_array.size == 768:
                        # Pad 768 to 3072
                        padded = np.pad(frame_array, (0, 3072-768), 'constant')
                        processed_frames.append(padded)
                    elif frame_array.size == 3072:
                        processed_frames.append(frame_array)
                    else:
                        print(f"Unexpected feature size at index {i}: {frame_array.size}")
                        processed_frames.append(np.zeros(3072))
                
                # Stack frames and handle sequence length
                sequence = np.vstack(processed_frames)
                if sequence.shape[0] < target_frames:
                    # Pad with zeros
                    padding = np.zeros((target_frames - sequence.shape[0], 3072), dtype=np.float32)
                    sequence = np.vstack([sequence, padding])
                elif sequence.shape[0] > target_frames:
                    # Truncate to first 8 frames
                    sequence = sequence[:target_frames]
                
                processed_x.append(sequence)
                
        except Exception as e:
            print(f"Error processing element {i}: {str(e)}")
            processed_x.append(np.zeros((target_frames, 3072), dtype=np.float32))
    
    ##print("Preprocessing complete!")
    return processed_x


def new_test_generator(num_features, new_test_x):
    # Preprocess the data first
    processed_x = preprocess_data(new_test_x)
    
    # For real-time inference, only one data point is passed at a time
    for i in range(len(processed_x)):
        # Prepare the input for a single point
        x_train = np.zeros((1, 8, num_features), dtype=np.float32)
        
        # Fill the input array with the processed data
        x_train[0] = processed_x[i]
        
        # Yield the single input point for inference
        return x_train

def xlnet_tokenize(sents, tokenizer):
    tok_sents = []
    for sen in sents:
        #print(f"Tokenizing Sentence: {sen}")  # Debug input sentence before tokenization
        tok_sents.append(tokenizer.tokenize(sen))
    #print(f"Tokenized Sentences: {tok_sents}")  # Debug output after tokenization
    return tok_sents

def sentence_marker(tokenized_sents):
    marker_array = []
    sent_num = 1
    for tokenized_sentence in tokenized_sents:
        sentence_marker = []
        #print(f"Processing Tokenized Sentence: {tokenized_sentence}")  # Debug current tokenized sentence
        for i in range(len(tokenized_sentence)):
            if i == 0:
                sentence_marker.append(sent_num)
            else:
                sentence_marker.append(0)
        #print(f"Sentence Marker for this sentence: {sentence_marker}")  # Debug sentence markers
        sent_num += 1
        marker_array.append(sentence_marker)
    
    #print(f"All Sentence Markers: {marker_array}")  # Debug all sentence markers
    return marker_array

def chunked_tokens_maker(all_toks, markers):
    splitted_toks = []
    splitted_markers = []
    l = 0
    r = 510
    while l < len(all_toks):
        #print(f"Chunking from index {l} to {r}")  # Debug the range of chunking
        splitted_toks.append(all_toks[l:min(r, len(all_toks))])
        splitted_markers.append(markers[l:min(r, len(markers))])
        l += 410
        r += 410
    
    #print(f"Chunked Tokens: {splitted_toks[:2]}")  # Debug first 2 chunks
    #print(f"Chunked Markers: {splitted_markers[:2]}")  # Debug first 2 chunk markers
    return splitted_toks, splitted_markers

def calculate_num_of_sents(chunk_marker_list):
    ct = 0
    for i in range(len(chunk_marker_list)):
        if chunk_marker_list[i] != 0:
            ct += 1
    ##print(f"Number of sentences in the chunk: {ct - 1}")  # Debug sentence count in chunk
    return ct - 1

def sentence_tokens_maker(marks, chunk_toks):
    pair_of_ids = []
    st = -1000
    ed = -1000
    for i, mark in enumerate(marks):
        if mark == -777:
            st = i
        if mark != -777 and mark != 777 and mark != 0:
            ed = i - 1
            pair_of_ids.append((st, ed))
            st = i
        if mark == 777:
            ed = i
            pair_of_ids.append((st, ed))
    
    ##print(f"Sentence Pairs: {pair_of_ids}")  # Debug sentence pairs
    return pair_of_ids

def att_masking(input_ids):
    attention_masks = []
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)
    ##print(f"Attention Masks: {attention_masks}")  # Debug attention masks
    return attention_masks


    

def get_output_for_one_vecL2(model,input_id, att_mask):
    device="cuda:0"
    input_ids = torch.tensor(input_id)
    att_masks = torch.tensor(att_mask)
    input_ids = input_ids.unsqueeze(0)
    att_masks = att_masks.unsqueeze(0)
    input_ids = input_ids.to(device)
    att_masks = att_masks.to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=att_masks)
    
    #print(f"Model Output: {outputs.logits}")  # Debug model output logits
    return outputs.logits

def get_XLNet_output_logits(encoded_sents, tokenizer, model):
    e_sents = []
    e_sents.append(encoded_sents)
    e_sents = pad_sequences(e_sents, maxlen=512, value=0, dtype="long", padding="pre")
    att_masks = att_masking(e_sents)
    
    #print(f"Encoded Sentences (after padding): {e_sents}")  # Debug padded encoded sentences
    logits = get_output_for_one_vecL2(model,e_sents[0], att_masks[0])
    #print(f"Logits for Encoded Sentences: {logits}")  # Debug logits for the encoded sentences
    return logits

    
def xlnet_detok(xtoks):
    #print(f"Tokens Before Detokenization: {xtoks}")  # Debug tokens before detokenization
    
    xtoks[0] = "_" + xtoks[0]
    words = []
    word = ""
    for tok in xtoks:
        if tok[0] == "▁":
            words.append(word)
            word = tok[1:]
        else:
            word += tok
    
    output = ""
    for word in words[1:]:
        output += (word + " ")

    #print(f"Detokenized Sentence: {output}")  # Debug the detokenized sentence
    return output

def get_explanation(model,chunked_xlnet_tokens, chunked_markers, chunk_scores, doc_num, tokenizer, predicted_label):
    explanation = ""
    
    for chunk_number, score in enumerate(chunk_scores[doc_num]):
        #print(f"Processing Chunk {chunk_number}, Score: {score}")  # Print current chunk number and its score
        
        # Check if chunk_number exceeds the number of chunks available in chunked_markers
        if chunk_number >= len(chunked_markers):
            #print(f"Skipping Chunk {chunk_number}: Out of Bounds")  # Debugging out-of-bounds chunks
            continue  # Skip this iteration if the chunk_number is out of bounds

        if chunk_number == 0:
            chunked_markers[chunk_number][0] = -777
            chunked_markers[chunk_number][-1] = 777
        else:
            if len(chunked_markers[chunk_number]) < 101:
                #print(f"Skipping Chunk {chunk_number}: Length < 101")  # Debug if chunk length is too small
                continue
            chunked_markers[chunk_number][100] = -777
            chunked_markers[chunk_number][-1] = 777

        if score < 0:
            #print(f"Skipping Chunk {chunk_number}: Negative Score")  # Debug if chunk score is negative
            continue

        ct_sent = calculate_num_of_sents(chunked_markers[chunk_number])
        #print(f"Chunk {chunk_number} - Number of Sentences: {ct_sent}")  # Print number of sentences per chunk

        top_k = 0.1 * ct_sent
        dict_sent_to_score = {}

        pair_of_ids = sentence_tokens_maker(chunked_markers[chunk_number], chunked_xlnet_tokens[chunk_number])
        #print(f"Chunk {chunk_number} - Sentence Pairs: {pair_of_ids[:3]}")  # Print first few sentence pairs

        CLS = tokenizer.cls_token
        SEP = tokenizer.sep_token
        PAD = tokenizer.pad_token

        original_logits = get_XLNet_output_logits(tokenizer.convert_tokens_to_ids(chunked_xlnet_tokens[chunk_number] + [SEP] + [CLS]), tokenizer, model)
        original_score = float(original_logits[0][predicted_label])
        ##print(f"Original Score for Chunk {chunk_number}: {original_score}")  # Print original score

        for i in range(len(pair_of_ids)):
            if pair_of_ids[i][0] == -1000:
                pair_of_ids[i] = (0, pair_of_ids[i][1]) 
            normalizing_length = pair_of_ids[i][1] - pair_of_ids[i][0] + 1
            if normalizing_length == 0:
                continue
            pad_sentence = [PAD] * normalizing_length

            left = chunked_xlnet_tokens[chunk_number][:pair_of_ids[i][0]]
            right = chunked_xlnet_tokens[chunk_number][pair_of_ids[i][1] + 1:]

            final_tok_sequence = left + pad_sentence + right + [SEP] + [CLS]
            encoded_sents = tokenizer.convert_tokens_to_ids(final_tok_sequence)
            logits = get_XLNet_output_logits(encoded_sents, tokenizer, model)
            score_for_predicted_label = float(logits[0][predicted_label])

            sent_score = 100

            if score_for_predicted_label > original_score:
                sent_score = -1 * (score_for_predicted_label - original_score)
            else:
                sent_score = original_score - score_for_predicted_label

            sent_score_norm = sent_score / normalizing_length
            sentence_in_words = xlnet_detok(chunked_xlnet_tokens[chunk_number][pair_of_ids[i][0]:pair_of_ids[i][1] + 1])
            dict_sent_to_score[sentence_in_words] = sent_score_norm

        sort_scores = sorted(dict_sent_to_score.items(), key=lambda x: x[1], reverse=True)
        #print(f"sort_scores are {sort_scores}")  # Print sorted sentences
        sorted_sentences = [i[0] for i in sort_scores]

        ##print(f"Sorted Sentences for Chunk {chunk_number}: {sorted_sentences[:3]}")  # Print sorted sentences

        for sentence in sorted_sentences[:int(top_k)]:
            explanation += sentence
            ###print(f"Added Sentence: {sentence}")  # Print each added sentence

    return explanation

### MAIN FUNCTION BEGINS HERE
''' MAIN CODE BELOW''' 

'''
nltk.download('punkt_tab')
nltk_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

Summary_tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
Summary_model = AutoModel.from_pretrained("law-ai/InLegalBERT")


modelDirectory="XLNet_FINAL"
tokenizer = XLNetTokenizer.from_pretrained(modelDirectory)
model = XLNetForSequenceClassification.from_pretrained(modelDirectory, output_hidden_states=True)
device="cuda:0"
model.to(device)


pdf_path = "Old_BANDHUA.PDF"
# Extract text and headnote from the PDF
extracted_text, headnote = extract_text_and_headnote_from_pdf(pdf_path)
print("Content extracted from PDF Succesfully")
print("="*100)

## LEVEL 1 Output 
embeds = generate_np_files_for_emb(model,extracted_text, tokenizer)
level1PathOutput="L1_Output/embeds.npy"
np.save(level1PathOutput, embeds)
print("LEVEL 1 EMBEDS GENERATED")
print("="*100)


### LEVEL 2 NOW

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
modelPred.load_weights('XGA_concat_epoch1_3.h5')
modelPred.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])

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
print("Case Final Prediction is:")
print(act_pred[0])
print("="*100)


### Case Explanation
chunk_scores = load("xlnet_occwts.npy", allow_pickle = True)
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
if act_pred[0]: ## If it is true, make prediction as 1
    pred_label=1
else:
    pred_label=0
explanation_of_this


'''