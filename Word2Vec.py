import re
import string
import gensim


def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


with open("Question1.txt", "r") as f:
    data = f.read()

# remove all html tags
data = remove_html_tags(data)
print("Tags Removed.")
# removing non-ascii characters
cleanText = re.sub(r'[^\x00-\x7f]', r'', data)
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

model = gensim.models.Word2Vec(sentences, size=100, window=4, min_count=5, workers=4)

print("Training Finished.")

word1 = model.wv["clean"]
word2 = model.wv["unclean"]
word3 = model.wv["amazed"]
word4 = model.wv["friendly"]

similarWords1 = model.wv.most_similar([word1], topn=10)
similarWords2 = model.wv.most_similar([word2], topn=10)
similarWords3 = model.wv.most_similar([word3], topn=10)
similarWords4 = model.wv.most_similar([word4], topn=10)

print("\nWords similar to 'clean':")
for sw in similarWords1:
    print(sw)

print("\nWords similar to 'unclean':")
for sw in similarWords2:
    print(sw)

print("\nWords similar to 'amazed':")
for sw in similarWords3:
    print(sw)

print("\nWords similar to 'friendly':")
for sw in similarWords4:
    print(sw)
