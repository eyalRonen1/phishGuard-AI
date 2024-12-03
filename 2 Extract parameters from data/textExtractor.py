import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog, Label, Entry, Button, Toplevel
import csv
import re
import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')  # For SentimentIntensityAnalyzer
from nltk import FreqDist
from nltk import pos_tag, ne_chunk, word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat.textstat import textstatistics
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from tkinter import filedialog, messagebox, scrolledtext, Tk, Label, Entry, Button, Frame
import extract_msg
import os

# Constants
CSV_FILENAME = "../textextractor/text_analysis_results.csv"

# Import string for string.punctuation
import string


def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens


def extract_bigrams(tokens):
    bigram_freq = FreqDist(ngrams(tokens, 2))
    most_common_bigrams = bigram_freq.most_common(5)
    return most_common_bigrams


def pos_tagging(tokens):
    pos_tags = pos_tag(tokens)
    return pos_tags


def named_entity_recognition(pos_tags):
    named_entities = ne_chunk(pos_tags)
    return named_entities


def sentiment_analysis(text):
    sentiment_analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = sentiment_analyzer.polarity_scores(text)
    return sentiment_scores['compound']


def readability_score(text):
    return textstatistics().flesch_reading_ease(text)


def count_numeric_chars(text):
    return sum(char.isdigit() for char in text)


def identify_ip_address(text):
    return bool(re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', text))


def calculate_lexical_diversity(text):
    words = text.split()
    word_count = len(words)
    unique_word_count = len(set(words))
    lexical_diversity = unique_word_count / word_count if word_count else 0
    return lexical_diversity


# Text analysis function
def analyze_text(text, url_email):
    # Tokenization
    tokens = tokenize_text(text)

    # Bigrams
    most_common_bigrams = extract_bigrams(tokens)
    bigrams_str = '|'.join([f"{' '.join(bigram)}:{freq}" for bigram, freq in most_common_bigrams])

    # POS Tagging
    pos_tags = pos_tagging(tokens)

    # Named Entity Recognition
    named_entities = named_entity_recognition(pos_tags)

    # Sentiment Analysis
    sentiment_score = sentiment_analysis(text)

    # Readability Scores (example: Flesch Reading Ease)
    readability_score_value = readability_score(text)

    # Analysis logic
    url_pattern = r'https?:\/\/(?:www\.)?[^\s\/]+(?:\/[^\s]*)?'
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    words = text.split()
    word_count = len(words)
    url_count = len(re.findall(url_pattern, url_email))  # Search only within the URL/email field
    email_count = len(re.findall(email_pattern, url_email))  # Search only within the URL/email field
    sentences = re.split(r'[.!?]+', text)
    sentence_count = len(sentences) - 1  # Adjust for the split behavior on final punctuation
    average_sentence_length = sum(
        len(sentence.split()) for sentence in sentences if sentence) / sentence_count if sentence_count else 0
    num_dots = text.count('.')
    url_length = len(url_email) if url_email else 'N/A'  # Assuming url_email contains a URL
    num_dash = text.count('-')
    at_symbol = '@' in text
    num_numeric_chars = count_numeric_chars(text)
    ip_address = identify_ip_address(text)
    punctuation_count = {char: text.count(char) for char in string.punctuation}
    lexical_diversity = calculate_lexical_diversity(text)

    # Package the results
    analysis_results = {
        "Word Count": word_count,
        "URL Count": url_count,
        "Email Count": email_count,
        "Sentence Count": sentence_count,
        "Average Sentence Length": average_sentence_length,
        "NumDots": num_dots,
        "UrlLength": url_length,
        "Most Common Bigrams": bigrams_str,
        "NumDash": num_dash,
        "AtSymbol": at_symbol,
        "NumNumericChars": num_numeric_chars,
        "IpAddress": ip_address,
        "Lexical Diversity": lexical_diversity,
        "POS Tags": '; '.join([f"{word}/{tag}" for word, tag in pos_tags]),
        "Named Entities": serialize_named_entities(named_entities),
        "Sentiment Score": sentiment_score,
        "Readability Score": readability_score_value,
    }

    # Include punctuation counts
    analysis_results.update(punctuation_count)

    return analysis_results


def serialize_named_entities(named_entities):
    serialized_entities = []
    for entity in named_entities:
        if hasattr(entity, 'label') and callable(entity.label):
            # Named entities are nested as trees
            entity_type = entity.label()
            entity_string = ' '.join(child[0] for child in entity)
            serialized_entities.append(f"{entity_type}: {entity_string}")
        else:
            # For tokens that are not named entities
            serialized_entities.append(f"{entity[0]}/{entity[1]}")
    return '; '.join(serialized_entities)


# Function to save analysis results to a CSV file
def save_results_to_csv(results):
    # Serialize complex data types
    results['Most Common Bigrams'] = results['Most Common Bigrams']
    # Ensure all other complex data types are serialized similarly

    # Check if file exists, if not, create it with headers
    try:
        with open(CSV_FILENAME, 'x', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=results.keys())
            writer.writeheader()
            writer.writerow(results)
    except FileExistsError:
        # Append the results to the CSV
        with open(CSV_FILENAME, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=results.keys())
            writer.writerow(results)

    messagebox.showinfo("Success", f"Analysis results appended to {CSV_FILENAME}.")

# GUI application class
class TextAnalysisApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Text Analysis Tool")
        self.geometry("800x500")  # Adjust the size as needed
        self.configure(background='light grey')  # Set a background color

        # Create a header label
        header_label = Label(self, text="Text Analysis Tool", font=("Helvetica", 16), bg='light grey')
        header_label.pack(pady=10)

        # Text input area with label
        Label(self, text="Input Text:", bg='light grey').pack(pady=(5, 0))
        self.text_input = scrolledtext.ScrolledText(self, height=10)
        self.text_input.pack(pady=5)
        # URL/Email input area with label
        Label(self, text="URL/Email address:", bg='light grey').pack(pady=(5, 0))
        self.url_email_input = Entry(self, width=70)
        self.url_email_input.pack(pady=5)

        # Button Frame
        button_frame = tk.Frame(self, bg='light grey')
        button_frame.pack(pady=10)

        # Analyze button
        analyze_button = Button(button_frame, text="Analyze", command=self.perform_analysis, bg='sky blue')
        analyze_button.grid(row=0, column=0, padx=5)

        # Save to CSV button
        save_button = Button(button_frame, text="Save to CSV", command=self.save_to_csv, bg='sky blue')
        save_button.grid(row=0, column=1, padx=5)

        # Load File button
        load_file_button = Button(button_frame, text="Load File", command=self.load_file, bg='sky blue')
        load_file_button.grid(row=0, column=2, padx=5)

    def load_file(self):
        filetypes = [("Text files", "*.txt"), ("Email files", "*.msg"), ("All files", "*.*")]
        file_path = filedialog.askopenfilename(filetypes=filetypes)
        if file_path:
            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension == '.msg':
                # Handle .msg file
                try:
                    with extract_msg.Message(file_path) as msg:
                        content = msg.body  # or msg.body if it's plain text
                        email_address = msg.sender  # Extract the sender email address
                        # Auto-fill the email address field
                        self.url_email_input.delete(0, 'end')
                        self.url_email_input.insert(0, email_address)
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load .msg file: {e}")
                    return
            else:
                # Handle other file types as before
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to read the file: {e}")
                    return

            # Clear and insert content into the text input area
            self.text_input.delete('1.0', 'end')
            self.text_input.insert('1.0', content)

    def perform_analysis(self):
        text = self.text_input.get("1.0", tk.END).strip()  # Strip whitespace to check for actual content
        if not text:  # If text is empty after stripping
            messagebox.showwarning("Empty Input", "The input is empty. Please insert text before continuing.")
            return  # Exit the method if the input is empty

        url_email = self.url_email_input.get()
        self.results = analyze_text(text, url_email)  # Continue with analysis if text is not empty

        # Display results in a new window with scrollable text
        result_window = tk.Toplevel(self)
        result_window.title("Analysis Results")
        result_text = scrolledtext.ScrolledText(result_window, wrap=tk.WORD)
        result_text.pack(expand=True, fill='both')

        # Construct a string of important results with explanations
        important_results = (
            f"Word Count: {self.results['Word Count']} (Total number of words)\n"
            f"URL Count: {self.results['URL Count']} (Total number of URLs found)\n"
            f"Email Count: {self.results['Email Count']} (Total number of email addresses found)\n"
            f"Sentence Count: {self.results['Sentence Count']} (Total number of sentences)\n"
            f"Average Sentence Length: {self.results['Average Sentence Length']:.2f} (Average number of words per sentence)\n"
            f"Lexical Diversity: {self.results['Lexical Diversity']:.2f} (Proportion of unique words to total words)\n"
            f"Sentiment Score: {self.results['Sentiment Score']} (Score indicating the emotional tone; closer to 1 is more positive)\n"
            f"Readability Score: {self.results['Readability Score']} (Flesch Reading Ease score; higher is easier to read)\n"
            f"NumDots: {self.results['NumDots']} (Number of periods in the text)\n"
            f"UrlLength: {self.results['UrlLength']} (Length of any URLs in characters)\n"
            f"NumDash: {self.results['NumDash']} (Number of dashes in the text)\n"
            f"AtSymbol: {'Present' if self.results['AtSymbol'] else 'Not present'} (Presence of @ symbol)\n"
            f"NumNumericChars: {self.results['NumNumericChars']} (Number of numeric characters in the text)\n"
            f"IpAddress: {'Present' if self.results['IpAddress'] else 'Not present'} (Presence of an IP address)\n"
            f"Most Common Bigrams: {self.results['Most Common Bigrams']} "
            "(The five most common two-word combinations and their frequencies)\n"
            # Add more explanations as needed
        )
        result_text.insert(tk.END, important_results)
        result_text.config(state=tk.DISABLED)

    def save_to_csv(self):
        if hasattr(self, 'results') and self.results:
            save_results_to_csv(self.results)
        else:
            messagebox.showerror("Error", "Please perform analysis before saving.")


# Run the application
if __name__ == "__main__":
    app = TextAnalysisApp()
    app.mainloop()