from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

#tweet we will be rating
tweet = 'That is great. Keep it up '

# preprocess our tweet
tweet_words = []

for word in tweet.split(' '):
    if word.startswith('@') and len(word) > 1:
        word = '@user'
    
    elif word.startswith('http'):
        word = "http"
    tweet_words.append(word)

tweet_process = " ".join(tweet_words)

# load model and tokenizer
roberta = "cardiffnlp/twitter-roberta-base-sentiment"

model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

labels = ['Negative', 'Neutral', 'Positive']

# sentiment analysis
encoded_tweet = tokenizer(tweet_process, return_tensors='pt')

output = model(**encoded_tweet)

#score outputs that we store in an array
scores = output[0][0].detach().numpy()
#softmax puts them into values between 0 and 1, so that they can be interpreted as probabilities
scores = softmax(scores)

for i in range(len(scores)):
    
    l = labels[i]
    s = scores[i]
    print(l,s)