import nltk

# complet example:
# https://simply-python.com/2014/03/14/saving-output-of-nltk-text-concordance/

f = open("text_file.txt")
raw = f.read()  # my raw text
tokenized = nltk.word_tokenize(raw)  # tokenize the raw text
text = nltk.Text(tokenized, name="document_text")  # create an NLTK text object
snippets = text.concordance("word")  # collect the snippets of occurence of the word "word"
