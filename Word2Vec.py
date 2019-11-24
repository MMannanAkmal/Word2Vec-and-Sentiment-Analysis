import re
import string

import gensim
from nltk import PorterStemmer


def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


ps = PorterStemmer()

with open("Question1.txt", "r") as f:
    data = f.read()

# remove all html tags
data = remove_html_tags(data)
print("Tags Removed.")
# removing non-ascii characters
cleanText = re.sub(r'[^\x00-\x7f]',r'', data)
print("Non-ASCII Characters Removed.")
# removing punctuation
regex = re.compile('[%s]' % re.escape(string.punctuation))
cleanText = re.sub(regex, '', cleanText)
print("Punctuation Removed.")
# converting to lowercase
cleanText = cleanText.lower()
print("Text Converted to LowerCase.")
# removing whitespace
cleanText = cleanText.strip()
print("Whitespace Removed.")
# split into sentences
lines = cleanText.split('\n')
print("Text split into sentences.")
# remove empty lines
lines = [i for i in lines if i]
print("Empty Lines Removed.")

sentences = []

for line in lines:
    sentence = line.split()
    sentences.append(sentence)

print("Training Started.")

model = gensim.models.Word2Vec(sentences, size=100, window=4, min_count=5, workers=1)

print("Training Finished.")

word1 = model.wv["clean"]
word2 = model.wv["unclean"]
word3 = model.wv["amazed"]
word4 = model.wv["friendly"]

similarWords1 = model.wv.most_similar([word1], topn=10)
similarWords2 = model.wv.most_similar([word2], topn=10)
similarWords3 = model.wv.most_similar([word3], topn=10)
similarWords4 = model.wv.most_similar([word4], topn=10)

print("Words similar to 'clean':")
for sw in similarWords1:
    print(sw)

print("Words similar to 'unclean':")
for sw in similarWords2:
    print(sw)

print("Words similar to 'amazed':")
for sw in similarWords3:
    print(sw)

print("Words similar to 'friendly':")
for sw in similarWords4:
    print(sw)

"""
---TEST OUTPUT---

Tags Removed.
Non-ASCII Characters Removed.
Punctuation Removed.
Text Converted to LowerCase.
Whitespace Removed.
Text split into sentences.
Empty Lines Removed.
Training Started.
Training Finished.
Words similar to 'clean':
('clean', 1.0)
('cleanthe', 0.8336902856826782)
('spotless', 0.795664370059967)
('immaculate', 0.7695684432983398)
('tidy', 0.7117076516151428)
('wellappointed', 0.7029688358306885)
('spacious', 0.6978803873062134)
('cleanall', 0.6881710290908813)
('welldecorated', 0.6758413910865784)
('roomy', 0.6613745093345642)
Words similar to 'unclean':
('unclean', 1.0000001192092896)
('dirty', 0.8078300952911377)
('grimy', 0.7992030382156372)
('dingy', 0.7957853674888611)
('grungy', 0.7914522290229797)
('stinky', 0.7748942971229553)
('grubby', 0.7741085290908813)
('unkempt', 0.7617216110229492)
('smelly', 0.7571333646774292)
('threadbare', 0.7568438649177551)
Words similar to 'amazed':
('amazed', 1.0)
('shocked', 0.8459294438362122)
('suprised', 0.8249524831771851)
('surprised', 0.8161122798919678)
('stunned', 0.8017778992652893)
('surpirsed', 0.7504333257675171)
('appalled', 0.7315530776977539)
('astonished', 0.7260863780975342)
('astounded', 0.7260578274726868)
('surprized', 0.7196154594421387)
Words similar to 'friendly':
('friendly', 1.0)
('courteous', 0.8912569284439087)
('polite', 0.8912397623062134)
('freindly', 0.8742184042930603)
('cordial', 0.8474617004394531)
('attentive', 0.8321975469589233)
('gracious', 0.8148832321166992)
('friendlythe', 0.812279462814331)
('curteous', 0.8113051652908325)
('accomodating', 0.790424644947052)

"""