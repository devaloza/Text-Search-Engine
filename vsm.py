# Import Module
import os
from xml.dom import minidom
from nltk.tokenize import word_tokenize
from datetime import datetime
import numpy as np
import json
from numpy import sqrt
from nltk.corpus import stopwords

#import xml.etree.ElementTree as ET
#parser = ET.XMLParser(encoding="utf-8")

# Folder Path
path = []
path = ["E:/DCU/CA6005_2022-20220126T235030Z-001/CA6005_2022/COLLECTION/COLLECTION", "E:/DCU/CA6005_2022-20220126T235030Z-001/CA6005_2022/topics/topics"]
totalLen = 0

final_topic = []
final_topic_tf = {}
final_collection = []
final_collection_tf = {}
new_final_collection_tf = {}
new_final_topic_tf = {}
query_vocab = []
coll_square_words = {}
topics_square_words = {}

def tf_function(temp, id):
    #print (temp)
    word_tf = {}
    tf_square_coll = 0
    for i in temp:
        global totalLen
        totalLen += len(temp)
        tf = temp.count(i) / len(temp)
        word_tf.update({
            i.lower() : tf 
        })
        tf_square_coll += tf*tf
    coll_square_words[id] = tf_square_coll
    return word_tf

def tf_function_topic(temp, id):
    #print (temp)
    word_tf = {}
    tf_square = 0
    for i in temp:
        tf = temp.count(i)
        word_tf.update({
            i.lower() : tf 
        })
        tf_square += tf*tf
    topics_square_words[id] = tf_square
    return word_tf

for dest in path:
 #   print (dest)
 #   print ("test")
    os.chdir(dest)
    def read_text_file(file_path):
        arr = {}
        with open(file_path, 'r') as f:
     #       print (file_path)
            xmldoc = minidom.parse(f)
            #print (file_path)
            if file_path.find('/topics/') != -1:
                #print ("id")
                if xmldoc.getElementsByTagName("QUERYID"):
                    qid = xmldoc.getElementsByTagName("QUERYID")[0]
                    if qid.firstChild:
                        arr['query_id'] = qid.firstChild.data
                if xmldoc.getElementsByTagName("TITLE"):
                    title = xmldoc.getElementsByTagName("TITLE")[0]
                    if title.firstChild:
                        arr['title']    = word_tokenize(title.firstChild.data)
                if xmldoc.getElementsByTagName("DESC"):
                    desc = xmldoc.getElementsByTagName('DESC')[0]
                    if desc.firstChild:
                        arr['desc']     = word_tokenize(desc.firstChild.data)
                return arr
            elif file_path.find('/COLLECTION/') != -1:
                #print ("elif")
                if xmldoc.getElementsByTagName("DOCID"):
                    docid = xmldoc.getElementsByTagName('DOCID')[0]
                    if docid.firstChild:
                        arr['docid']     = docid.firstChild.data
                if xmldoc.getElementsByTagName("HEADLINE"):
                    headline = xmldoc.getElementsByTagName('HEADLINE')[0]
                    if headline.firstChild:
                        arr['headline']     = word_tokenize(headline.firstChild.data)
                if xmldoc.getElementsByTagName("TEXT"):
                    text = xmldoc.getElementsByTagName('TEXT')[0]
                    if text.firstChild:
                        arr['text']     = word_tokenize(text.firstChild.data)
                return arr
# iterate through all file
    start = datetime.now()
    print ("imposrt start -->", start)
    i = 0
    #print (os)
    for file in os.listdir():
        file_path = f"{dest}\{file}"
        
        # print(file_path)
        
        # call read text file function
        result = read_text_file(file_path)
        #print (result)
        #print ("----------------------")
        if "query_id" in result.keys():
            #print (result['title'])
            #print (result['desc'])
            #break
            temp = result['title'] + result['desc']
            for tex in temp:
                if tex.lower() not in query_vocab:
                    query_vocab.append(tex.lower())
            final_topic.append(result['title'] + result['desc'])
            # final_topic_tf[i] = tf_function(temp)
            final_topic_tf.update({
                #result['docid']:result['headline'] + result['text']
                result['query_id']:tf_function_topic(temp, result['query_id'])
                #tf_function(temp)
            })
        elif "docid" in result.keys():
            if "headline" in result and "text" in result:
                final_collection.append([word.lower() for word in result['headline']] + [word2.lower() for word2 in result['text']])
                temp = result['headline'] + result['text']
                final_collection_tf.update({
                    #result['docid']:result['headline'] + result['text']
                    result['docid']:tf_function(temp, result['docid'])
                })
fc = final_collection
#print (topics_square_words)
#print (coll_square_words)
#exit()
#print (fc)
print ("imposrt end -->", datetime.now())
ft = final_topic
docf = {}

for key, vals in final_collection_tf.items():
    for j in set(vals):
        #freq = vals.count(j)
        #print (j)
        if j not in docf:
            docf.update({
                j:1
            })
        else:
            docf.update({
                j: docf[j] + 1
            })
# print (docf)
# exit()
for j, vals in final_collection_tf.items():
 #   print (j)
    # vals = final_collection_tf[j]
    for k in vals:
        idf = np.log( totalLen / docf[k])
        new_final_collection_tf.update({
            k:round(vals[k]*idf, 4)
        })
        final_collection_tf.update({
            j:new_final_collection_tf
        })
end = datetime.now()
print ("end-->", end)

print ("similarity start -->", datetime.now())
#(final_collection_tf, final_topic_tf)

common_docs_word = {}
common_topic_docs_word = {}
for term, val in final_topic_tf.items():
    for val1, val2 in val.items():
        #print (val1)
        common_words = []
        for j in final_collection_tf:
            #print (final_collection_tf[j])
            if val1 in final_collection_tf[j] and val1 not in common_words:
                common_words.append(val1)
            common_docs_word.update({
                j:common_words
            })
    common_topic_docs_word.update({
        term:common_docs_word
    })
final_topic_collection_cosion = {}
final_collection_cosine = {}
for words in common_topic_docs_word:
    numerator = 0
    final_score = 0
    for common_word in common_topic_docs_word[words]:
        current_word = common_topic_docs_word[words][common_word]
        for i in current_word:
            if i in final_topic_tf[words]:
                numerator += final_collection_tf[common_word][i] * final_topic_tf[words][i]
                final_score += numerator / (sqrt(coll_square_words[common_word]) * sqrt(topics_square_words[words]))
        #print (common_topic_docs_word[words][common_word])
        #exit()
            final_collection_cosine.update({
                common_word:final_score
            })
    final_topic_collection_cosion.update({
        words:final_collection_cosine
    })


final_scores = []
#(final_collection_tf, final_topic_tf)
# qur_tfidf = final_topic_tf
# doc_tfidf = final_collection_tf

# for q_key, query in qur_tfidf.items():
#     for doc_key, document in doc_tfidf.items():

#         numerator = 0
#         denominator_sum_square_doc = 0
#         denominator_sum_square_query = 0
#         for word in query:
#             if word in document:
#                 numerator += query[word] * document[word]

#         for word_tfidf in document.values():
#             denominator_sum_square_doc += word_tfidf ** word_tfidf

#         for query_tf in query.values():
#             denominator_sum_square_query += query_tf ** query_tf

#         score = numerator / sqrt(denominator_sum_square_doc*denominator_sum_square_query)

#         final_scores.append(f"{q_key} Q0 {doc_key} 0 {score} nop\n")
# with open('E:/DCU/CA6005_2022-20220126T235030Z-001/CA6005_2022/vsm.txt', 'w') as w:
#     for j in final_scores:
#         w.writelines(j)
# #print (final_scores)
# exit()

#print (final_topic_collection_cosion)
with open('E:/DCU/CA6005_2022-20220126T235030Z-001/CA6005_2022/vsm.txt', 'w') as w:
    for query_id in final_topic_collection_cosion:
        for doc_id in final_topic_collection_cosion[query_id]:
            # print (doc_id)
            # print (query_id)
            # print (final_topic_collection_cosion[query_id][doc_id])
            w.writelines(f"{query_id} Q0 {doc_id} 0 {final_topic_collection_cosion[query_id][doc_id]} nop\n")
print ('script end', datetime.now())            
exit()
def similarity_cosin(doc_tfidf, qur_tfidf):
 #   print (doc_tfidf)
    similarity_matrix = dict()
    final_output_similar_matrix = []
    numerator = 0
    denominator_a = 0
    denominator_b = 0
    vsmoutput = {}
    for term, val in qur_tfidf.items():
        score = 0
        vsmdocscore = [0]*len(doc_tfidf)
        # print (val)
        for val1, val2 in val.items():
            numerator = 0
            for i, (document, value) in enumerate(doc_tfidf.items()):
                #print (document)
                #print (value)
                if val1 not in doc_tfidf[document]:
                    continue
                numerator += doc_tfidf[document][val1] * val2
                vsmdocscore[i] += numerator / (sqrt(coll_square_words[document]) * sqrt(topics_square_words[term]))
            #final_output_similar_matrix.append(f"{term} Q0 {document} 0 {score} nop")
        vsmoutput.update({
            term:vsmdocscore
        })
    return vsmoutput # Vector Space Output 
print (final_output)
print ("similarity end -->", datetime.now())