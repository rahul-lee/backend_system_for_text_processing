```markdown
# Flask Text Processing API

This repository contains a Flask-based RESTful API for processing text input. The API performs various operations, including content moderation, sentiment analysis, and text summarization using OpenAI's GPT-4. It also includes robust error handling, rate limiting, and logging.

---

## Features

- **Content Moderation**: Detects and rejects offensive content using a pre-trained `toxic-bert` model.
- **Sentiment Analysis**: Classifies text as Positive, Negative, or Neutral.
- **Text Summarization**: Provides concise summaries of input text using OpenAI's GPT-4.
- **Rate Limiting**: Limits API usage to prevent abuse (3 requests per minute per IP, 200 requests per day).
- **Logging**: Captures events and errors for debugging and monitoring.
- **Health Check**: Verifies connectivity to the OpenAI API.
- **History Retrieval**: Maintains an in-memory log of processed results.

---

## Prerequisites

1. Python 3.8 or above.
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. OpenAI API key saved in `config.py`:
   ```python
   TOKEN = "your-openai-api-key"
   ```

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name/text-processing-api.git
   cd text-processing-api
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the server:
   ```bash
   python app.py
   ```

The server will start at `http://127.0.0.1:5000/`.

---

## Endpoints

### 1. **Process Text**
   - **URL**: `/process`
   - **Method**: `POST`
   - **Description**: Processes the input text for moderation, sentiment analysis, and summarization.
   - **Payload**:
     ```json
     {
       "text": "Your text to process"
     }
     ```
   - **Response**:
     ```json
     {
       "processed_text": "Summary of the text",
       "category": "Positive Content",
       "sentiment": "POSITIVE"
     }
     ```

### 2. **Get History**
   - **URL**: `/history`
   - **Method**: `GET`
   - **Description**: Retrieves the history of processed texts.
   - **Response**:
     ```json
     {
       "history": [
         {
           "input_text": "Your input text",
           "processed_text": "Summary",
           "category": "Positive Content",
           "sentiment": "POSITIVE"
         }
       ]
     }
     ```

### 3. **Health Check**
   - **URL**: `/health`
   - **Method**: `GET`
   - **Description**: Checks the health of the OpenAI API.
   - **Response**:
     ```json
     {
       "status": "OK",
       "openai_status": "reachable"
     }
     ```

---

## Configuration

- **Rate Limits**:
  Modify limits in the `Limiter` setup:
  ```python
  default_limits=["3 per minute", "200 per day"]
  ```

- **Logging Level**:
  Adjust the logging level as needed:
  ```python
  logging.basicConfig(level=logging.INFO)
  ```

---

## Error Handling

The API handles various errors gracefully and provides detailed responses for debugging:
- **Invalid Input**: `400 Bad Request`
- **Offensive Content**: `400 Bad Request`
- **Rate Limit Exceeded**: `429 Too Many Requests`
- **Authentication Error**: `401 Unauthorized`
- **OpenAI API Error**: `500 Internal Server Error`
- **Unexpected Errors**: `500 Internal Server Error`

---

## Dependencies

- Flask
- Flask-Limiter
- OpenAI
- Transformers
- Logging

Install all dependencies with:
```bash
pip install flask flask-limiter openai transformers
```

---

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to improve this repository.

---

## Acknowledgments

- [OpenAI](https://openai.com/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
```
