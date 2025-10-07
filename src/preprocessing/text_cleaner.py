from src.exception.exception import FeedException
from src.logging.logger import logging
import re
import html
from typing import Optional
import string


class TextCleaner:
    def __init__(self, preserve_hashtag_text: bool = True, remove_emojis: bool = True):
        self.preserve_hashtag_text = preserve_hashtag_text
        self.remove_emojis = remove_emojis
            
        self.stopwords = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "to", "of",
            "in", "for", "on", "and", "or", "as", "it", "that", "this", "with", "at",
            "by", "from", "but", "if", "so", "then", "there", "here", "which", "when",
            "what", "who", "whom", "do", "does", "did", "done", "can", "could", "would",
            "should", "will", "just", "than", "about", "into", "over", "under", "again",
            "further", "up", "down", "out", "off", "once", "only", "very", "now",
            "very", "still", "also", "even", "much", "most", "some", "any", "all",
            "both", "few", "more", "other", "another", "well", "say", "new", "good",
            "first", "last", "two", "three", "four", "five", "six", "seven", "eight",
            "nine", "ten", "one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine", "ten"  
        }
        
    def _remove_urls(self, text):
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, "", text, flags=re.MULTILINE | re.IGNORECASE)
        
        
    def _remove_emails(self, text):
        return re.sub(r"\S+@\S+\.\S+", "", text)
        
        
    def _remove_html(self, text):
        text = html.unescape(text)
        text = re.sub(r"<.*?>", "", text)
            
        text = re.sub(r"<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>", "", text, flags=re.IGNORECASE)
        return text
        
        
    def _handle_hashtags(self, text):
        if self.preserve_hashtag_text:
            return re.sub(r"#(\w+)", r"\1", text)
        return re.sub(r"#\w+", "", text)
        
    
    def _remove_mentions(self, text):
        return re.sub(r"@\w+", "", text)
        
        
    def _remove_emojis_func(self,text):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   u"\U0001F918-\U0001F919"  # new emojis
                                   u"\U0001F680-\U0001F6FF"  # transport
                                   u"\U0001F700-\U0001F77F"  # alchemical
                                   u"\U0001F780-\U0001F7FF"  # geometric shapes extended
                                   u"\U0001F800-\U0001F8FF"  # supplemental arrows-c
                                   u"\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
                                   u"\U0001FA00-\U0001FA6F"  # chess symbols
                                   u"\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-a
                                   u"\U00002702-\U000027B0"  # dingbats
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
            
        return emoji_pattern.sub(r'', text)
        
        
    def _remove_punctuation_numbers(self,text):
        text = re.sub(r"\b\d+\b", "", text)
        punctuation_to_remove = string.punctuation.replace('!', '').replace('?', '').replace('.', '').replace(',', '')
        text = text.translate(str.maketrans("","" , punctuation_to_remove))
        return text
    
    def _remove_stopwords(self, text: str) -> str:
        words = text.split()
        # Improved: Preserve words with apostrophes for contractions/possessives
        words = [w for w in words if w.lower() not in self.stopwords and "'" in w or len(w) > 2]
        return " ".join(words)
        
        
    def _expand_contraction(self, text):
        contractions_dict = {
                    "ain't": "is not", "aren't": "are not", "can't": "cannot", "couldn't": "could not",
                    "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
                    "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will",
                    "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will",
                    "how's": "how is", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",
                    "i'll've": "i will have", "i'm": "i am", "i've": "i have", "isn't": "is not",
                    "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have",
                    "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not",
                    "mightn't": "might not", "might've": "might have", "mustn't": "must not",
                    "must've": "must have", "needn't": "need not", "not've": "not have",
                    "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
                    "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                    "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
                    "she'll've": "she will have", "she's": "she is", "shouldn't": "should not",
                    "should've": "should have", "so've": "so have", "so's": "so as",
                    "that'd": "that would", "that'd've": "that would have", "that's": "that is",
                    "there'd": "there would", "there'd've": "there would have", "there's": "there is",
                    "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
                    "they'll've": "they will have", "they're": "they are", "they've": "they have",
                    "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have",
                    "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have",
                    "weren't": "were not", "what'll": "what will", "what'll've": "what will have",
                    "what're": "what are", "what's": "what is", "what've": "what have",
                    "when's": "when is", "when've": "when have", "where'd": "where did",
                    "where's": "where is", "where've": "where have", "who'll": "who will",
                    "who'll've": "who will have", "who's": "who is", "who've": "who have",
                    "why's": "why is", "why've": "why have", "will've": "will have",
                    "won't": "will not", "won't've": "will not have", "would've": "would have",
                    "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                    "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are",
                    "y'all've": "you all have", "you'd": "you would", "you'd've": "you would have",
                    "you'll": "you will", "you'll've": "you will have", "you're": "you are",
                    "you've": "you have"
                }
            
        for contraction, expansion in contractions_dict.items():
            text = re.sub(re.escape(contraction), expansion,text, flags=re.IGNORECASE)
        return text
        
        
    def clean(self, text):
        if not text or not isinstance(text,str):
            logging.warning("Invalid text input provided for cleaning.")
            return ""
        
        original_length = len(text)
        text = text.lower()
        text = self._expand_contraction(text)
        text = self._remove_urls(text)
        text = self._remove_emails(text)
        text = self._remove_html(text)
        text = self._handle_hashtags(text)
        text = self._remove_mentions(text)
        if self.remove_emojis:
            text = self._remove_emojis_func(text)
        text = self._remove_punctuation_numbers(text)
        text = self._remove_stopwords(text)
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) == 0:
            logging.debug(f"Text became empty after cleaning. Original length: {original_length}")
        return text