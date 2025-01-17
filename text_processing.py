import re
from flask import Flask, request, jsonify
import openai
import logging
from openai import OpenAIError, RateLimitError, AuthenticationError
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from transformers import pipeline
from config import TOKEN  # Importing API token securely from config.py
import time

# Initialize OpenAI client using the API key from the configuration file
client = openai.OpenAI(api_key=TOKEN)

# Initialize Flask app
app = Flask(__name__)

# Set up logging to capture events and errors
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up rate limiting for the app (limits the number of requests per minute/day)
limiter = Limiter(
    get_remote_address,  # Using the client's IP address to apply rate limiting
    app=app,
    default_limits=["3 per minute", "200 per day"]
)

# Define the maximum input length (in tokens) for text input
MAX_INPUT_LENGTH = 2048

# Load pre-trained models for content moderation (toxic content detection) and sentiment analysis
moderator = pipeline("text-classification", model="unitary/toxic-bert")
sentiment_analyzer = pipeline("sentiment-analysis")

# In memory storage to keep track of processed results for future retrieval
processed_results = []

# Function to categorize text based on sentiment analysis results
def categorize_text(sentiment):
    if sentiment == 'POSITIVE':
        return 'Positive Content'
    elif sentiment == 'NEGATIVE':
        return 'Negative Content'
    else:
        return 'Neutral Content'

# Endpoint to process the text input for moderation, sentiment analysis, and summarization
@app.route('/process', methods=['POST'])
@limiter.limit("3 per minute")  # Apply rate limit for this endpoint
def process_text():
    try:
        # Get JSON data from the incoming request
        data = request.get_json()

        # Validate input, check if 'text' key exists and is not empty
        if not data or 'text' not in data or not data['text'].strip():
            logger.warning("Invalid input: Text input cannot be empty.")
            return jsonify({"error_code": "INVALID_INPUT", "error_message": "Text input cannot be empty."}), 400

        input_text = data['text']

        # Check the length of input text to prevent exceeding token limit
        if len(input_text) > MAX_INPUT_LENGTH:
            logger.warning(f"Input text exceeds maximum length of {MAX_INPUT_LENGTH} tokens.")
            return jsonify({"error_code": "INPUT_TOO_LONG", "error_message": f"Text input exceeds maximum length of {MAX_INPUT_LENGTH} tokens."}), 400

        # Check for offensive content using the pre trained content moderation model
        moderation_result = moderator(input_text)
        if any(result['label'] == 'toxic' and result['score'] > 0.5 for result in moderation_result):
            logger.warning("Input text contains offensive content.")
            return jsonify({"error_code": "OFFENSIVE_CONTENT", "error_message": "Text input contains offensive content."}), 400

        # Perform sentiment analysis to classify the text as positive, negative, or neutral
        sentiment_result = sentiment_analyzer(input_text)[0]['label']

        # Categorize the text based on the sentiment analysis result
        category = categorize_text(sentiment_result)

        # Perform summarization using OpenAI GPT-4 (retry logic in case of rate limit exceeded)
        for attempt in range(3):  # Retry up to 3 times
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",  # Use optimized GPT-4 model for summarization
                    store=True,
                    messages=[{"role": "user", "content": f"Summarize the following text:\n{input_text}"}],
                    max_tokens=50,  # Limit the summarization output to 50 tokens
                    temperature=0.7  # Set randomness level for generation
                )
                break  # Exit loop if the request succeeds
            except RateLimitError as rle:
                logger.error(f"Rate limit exceeded: {str(rle)}")
                if attempt < 2:  # Retry after 1 second
                    time.sleep(1)
                else:
                    return jsonify({"error_code": "RATE_LIMIT_EXCEEDED", "error_message": f"Rate limit exceeded: {str(rle)}"}), 429

        # Process the summary result from OpenAI response
        result = response.choices[0].message['content'].strip()
        logger.info("Text processed successfully.")

        # Store the processed result (input text, summary, category, sentiment) in the in-memory storage
        processed_results.append({
            "input_text": input_text,
            "processed_text": result,
            "category": category,
            "sentiment": sentiment_result
        })

        # Return the processed text, category, and sentiment in the response
        return jsonify({"processed_text": result, "category": category, "sentiment": sentiment_result}), 200
    except RateLimitError as rle:
        logger.error(f"Rate limit exceeded: {str(rle)}")
        return jsonify({"error_code": "RATE_LIMIT_EXCEEDED", "error_message": f"Rate limit exceeded: {str(rle)}"}), 429
    except AuthenticationError as ae:
        logger.error(f"Authentication error: {str(ae)}")
        return jsonify({"error_code": "AUTHENTICATION_ERROR", "error_message": f"Authentication error: {str(ae)}"}), 401
    except OpenAIError as oe:
        # Handle OpenAI API related errors with a descriptive message
        logger.error(f"OpenAI API error: {str(oe)}")
        return jsonify({
            "error_code": "OPENAI_API_ERROR",
            "error_message": str(oe),
            "openai_error_message": getattr(oe, 'message', 'No specific message'),
            "openai_error_type": getattr(oe, 'type', 'Unknown type')
        }), 500
    except Exception as e:
        # Catch unexpected errors and log them
        logger.error(f"An unexpected error occurred: {str(e)}")
        return jsonify({"error_code": "UNEXPECTED_ERROR", "error_message": f"An unexpected error occurred: {str(e)}"}), 500

# Endpoint to retrieve the history of processed results
@app.route('/history', methods=['GET'])
def get_history():
    return jsonify({"history": processed_results}), 200

# Health check endpoint to verify if the OpenAI API is reachable
@app.route('/health', methods=['GET'])
def health_check():
    try:
        # Make a quick ping to OpenAI API to check its status
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            store=False,
            messages=[{"role": "system", "content": "Health check"}],
            max_tokens=1  # Minimal output to keep the check lightweight
        )
        return jsonify({"status": "OK", "openai_status": "reachable"}), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({"status": "degraded", "error_message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
