import re
import emoji
import numpy as np 
import pandas as  pd
import seaborn as sns
import sys, re, argparse
from konlpy.tag import Okt
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib import font_manager, rc
from gensim.models.fasttext import FastText
from soynlp.normalizer import repeat_normalize

# sns 한국어 패치
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)


def text_preprocess(text):
    pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣]+')
    url_pattern = re.compile(
        r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
 
    text = pattern.sub(' ', text)
    text = emoji.replace_emoji(text, replace='') #emoji 삭제
    text = url_pattern.sub('', text)
    text = text.strip()
    text = repeat_normalize(text, num_repeats=2)
    
    return text


def embedding_sentence(model_path, df, embedding_dim=100 ,length_limit=None):
    
    model = FastText.load(model_path)
    text_list = df['text']
    class_list = df['class']
    
    empty_list = []
    # 1. text 전처리 
    for idx, text in enumerate(text_list):
        text_list[idx] = text_preprocess(text)
        if text_list[idx] == "":
            empty_list.append(idx)
    
    # empty_list에 해당하는 인덱스 제외한 text_list와 class_list 재정의
    text_list = [text_list[idx] for idx in range(len(text_list)) if idx not in empty_list]
    class_list = [class_list[idx] for idx in range(len(class_list)) if idx not in empty_list]
    df = df.drop(empty_list)

    # 2. text tokenize
    tokenizer = Okt()
    tokenized_text_list = []
    for cleaned_text in text_list:
        tokens = tokenizer.morphs(cleaned_text)
        tokenized_text_list.append(tokens)
    df['tokens'] = tokenized_text_list
    
    # 3. 만일 길이 제한 걸었으면 변해야함
    if length_limit:
        # tokenized_text_list에서 길이가 length_limit 이하인 항목들만 필터링
        filtered_indices = [idx for idx, tokens in enumerate(tokenized_text_list) if len(tokens) <= length_limit]

        # 필터링된 인덱스에 해당하는 행들만 남김
        df = df.iloc[filtered_indices]
    
    # 4. df에서 문장들 각각 matrix 로 변환  
    token_list = df['tokens']
    class_list = df['class']
    

    size = len(token_list)
    matrix = np.zeros((size, embedding_dim))
    
    for idx, token in enumerate(token_list):
        vector = np.array([
            model.wv[t] for t in token
        ])

        final_vector = np.mean(vector, axis = 0)
        matrix[idx] = final_vector
        
    return matrix, class_list


def visulize_sentences(title ,matrix, class_list, class_dict=None):
    
    vectors = matrix # 여기에 100차원 벡터 데이터를 입력하세요
    classes = class_list # 여기에 해당 벡터의 클래스 라벨(0에서 10 사이의 값)을 입력하세요
    
    if class_dict is not None:
        for idx, tmp_class in enumerate(classes):
            classes[idx] = class_dict[tmp_class]
        
    class_kind = len(set(classes))
    # t-SNE 임베딩 수행
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    vectors_tsne = tsne.fit_transform(vectors)

    # 시각화
    plt.figure(figsize=(20, 10))
    sns.scatterplot(
        x=vectors_tsne[:, 0], y=vectors_tsne[:, 1],
        hue=classes,
        palette=sns.color_palette("hsv", class_kind),
        legend="full",
        alpha=0.7
    )

    plt.title(title)
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')