# Import Module
import os
from xml.dom import minidom
from nltk.tokenize import word_tokenize
from datetime import datetime
import numpy as np
import json
from numpy import sqrt
import xml.etree.ElementTree as ET
import xmltodict


path = ["E:/DCU/CA6005_2022-20220126T235030Z-001/CA6005_2022/COLLECTION/COLLECTION"]
path2 = ["E:/DCU/CA6005_2022-20220126T235030Z-001/CA6005_2022/topics/topics"]
final_collection = []
final_topic = {}
i = 0
arr = {}
arr1 = {}
total_len = 0
final_collection_withid = {}
print ('import start')
start = datetime.now()
print ("start -->", start)
file_write = open('E:/DCU/CA6005_2022-20220126T235030Z-001/CA6005_2022/collection.txt', 'w')
file_write_topic = open('E:/DCU/CA6005_2022-20220126T235030Z-001/CA6005_2022/topics.txt', 'w')
for dest in path:
    os.chdir(dest)
    for file in os.listdir():
            file_path = f"{dest}\{file}"
            with open(file_path, 'r') as f:
                file_write.write(file_path +"\n")
                xmldoc = minidom.parse(f)
                if xmldoc.getElementsByTagName("DOCID"):
                    docid = xmldoc.getElementsByTagName('DOCID')[0]
                    if docid.firstChild:
                        arr['docid']     = docid.firstChild.data
                if xmldoc.getElementsByTagName("TEXT"):
                    title = xmldoc.getElementsByTagName("TEXT")[0]
                    #print (title)
                    if title.firstChild:
                        arr['text']    = word_tokenize(title.firstChild.data)
                    if xmldoc.getElementsByTagName("HEADLINE"):
                        desc = xmldoc.getElementsByTagName('HEADLINE')[0]
                        if desc.firstChild:
                            arr['headline']     = word_tokenize(desc.firstChild.data)
                    total_len += len(arr['text']) + len(arr['headline'])
                    result = arr
                    final_collection.append([word.lower() for word in result['headline']] + [word2.lower() for word2 in result['text']])
                    final_collection_withid.update({
                    #result['docid']:result['headline'] + result['text']
                        result['docid']:[word1.lower() for word1 in arr['text']] + [word3.lower() for word3 in arr['headline']]
                    })
for dest in path2:
    os.chdir(dest)
    for file in os.listdir():
            file_path = f"{dest}\{file}"
            with open(file_path, 'r') as f:
                file_write_topic.write(file_path +"\n")
                xmldoc = minidom.parse(f)
                #print (xmldoc)
                if xmldoc.getElementsByTagName("QUERYID"):
                    query_id = xmldoc.getElementsByTagName("QUERYID")[0]
                    #print (query_id)
                    if query_id.firstChild:
                        arr1['query_id']    = (query_id.firstChild.data)
                if xmldoc.getElementsByTagName("TITLE"):
                    title = xmldoc.getElementsByTagName("TITLE")[0]
                    if title.firstChild:
                        arr1['title']    = word_tokenize(title.firstChild.data)
                if xmldoc.getElementsByTagName("DESC"):
                    desc = xmldoc.getElementsByTagName('DESC')[0]
                    if desc.firstChild:
                        arr1['desc']     = word_tokenize(desc.firstChild.data)
                result = arr1
                #print (result)
                final_topic.update({
                    result['query_id']:[word.lower() for word in result['title']] + [word2.lower() for word2 in result['desc']]
                })
print ('import end')

endnow = datetime.now()
print ("end -->", endnow)

start = datetime.now()
print ("start -->", start)

#print (final_topic)
#print (total_len)
avgdl = sum(len(sentence) for sentence in final_collection) / len(final_collection)
N = len(final_collection)
#print (avgdl)
#print (N)

collection_tf = {}
docf = {}
docidf = {}
print ('total frequence count start', datetime.now())
for key, vals in final_collection_withid.items():
    for j in set(vals):
        freq = vals.count(j)
        #print (j)
        if j not in docf:
            docf.update({
                j:1
            })
        else:
            docf.update({
                j: docf[j] + 1
            })
        collection_tf.update({
            j : freq
        })
    final_collection_withid.update({
        key:collection_tf
    })
print ('document freq calculated end..', datetime.now())

for i in docf:
    docidf[i] = np.log(((N - docf[i] + 0.5) / (docf[i] + 0.5)) + 1)
#print (docidf)

#print (final_topic)
#print (final_collection_withid)
print ('idf calcolated..', datetime.now())
bm25output = {}

for queryId, query in final_topic.items():
    bm25docscore = [0]*len(final_collection_withid)
    for word in query:
        #print (word+"\n")
        for i, (k, val) in enumerate(final_collection_withid.items()):  
            score = 0  
            if word not in val:
                continue
            freq = val[word]
            tf = (freq * (1.2 + 1)) / (freq + 1.2 * (1 - 0.75 + 0.75 * (len(val) / avgdl)))
            bm25docscore[i] += round(tf*docidf[word], 4)
    bm25output.update({
        queryId:bm25docscore
    })

with open('E:/DCU/CA6005_2022-20220126T235030Z-001/CA6005_2022/bm25.txt', 'w') as w:
    for i in bm25output:
        for j in zip(bm25output[i], list(final_collection_withid.keys())):
            w.writelines(f"{i} Q0 {j[1]} 0 {j[0]} nop\n")
end = datetime.now()
print ("end -->", end)