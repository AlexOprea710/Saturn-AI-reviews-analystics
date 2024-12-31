import spacy
from transformers import pipeline
import nltk

nltk.download('stopwords')

sentiment_pipeline = pipeline("sentiment-analysis")

def read_data(file_path):
    """Citește recenziile din fișierul dat"""
    with open(file_path, 'r') as file:
        reviews = file.readlines()
        return reviews
    
def analyze_sentiment(reviews):
        """Analizează sentimentul recenziilor"""
        for review in reviews:
            result = sentiment_pipeline(review)
            print(f"Review: {review.strip()} => Sentiment: {result[0]['label']} with score {result[0]['score']}")

def main():
    reviews = read_data('data.txt')
  
    analyze_sentiment(reviews)

if __name__ == '__main__':
    main()