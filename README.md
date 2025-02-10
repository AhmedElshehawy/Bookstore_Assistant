# Bookstore Assistant

![Bookstore Chatbot](imgs/bookstore_assistant.png)

You can try the chatbot [here](https://huggingface.co/spaces/elshehawy/BookBot)

## Overview

Bookstore Assistant is a key component of a larger system consisting of four interconnected repositories. Together, these repositories enable users to chat with a comprehensive books database through an intelligent chatbot interface. This repository specifically handles the chatbot interactions, processing user queries, and routing them to retrieve relevant book data.

## System Architecture

The complete system is divided into four specialized repositories:

- **Bookstore Assistant**: Manages the chat interface, processes natural language queries, and interacts with other components.
- [**Bookstore DB**](https://github.com/AhmedElshehawy/Bookstore-DB): Responsible for storing and querying book database.
- [**Bookstore Scraper**](https://github.com/AhmedElshehawy/Bookstore_Scraper): Scrapes data from the web and stores it in the DB.
- [**Bookstore Frontend (BookBot)**](https://huggingface.co/spaces/elshehawy/BookBot): A chat interface for the chatbot.

Each repository plays a unique role, and together they provide a seamless, secure, and efficient way for users to obtain book information.

## Features

- **Natural Language Processing**: Converts user queries into structured SQL queries.
- **SQL Query Generation & Safety**: Generates optimized SQL queries while ensuring they are safe for execution.
- **Chat History Management**: Stores conversation histories using DynamoDB.
- **Robust Integration**: Connects to a PostgreSQL database that stores comprehensive book details.
- **Modular Architecture**: Part of a multi-repo system that promotes separation of concerns and easier maintainability.

## Technologies Used

- [Python](https://www.python.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [PostgreSQL](https://www.postgresql.org/)
- [DynamoDB](https://aws.amazon.com/dynamodb/)
- [LangGraph & LangChain](https://github.com/langchain-ai/)
- [OpenAI API](https://platform.openai.com/)
- [AWS Lambda](https://aws.amazon.com/lambda/)
- [Mangum](https://mangum.io/)

## Installation

1. **Clone the Repository**

   ```bash
   git clone <repository-url>
   ```

2. **Navigate to the Project Directory**

   ```bash
   cd Bookstore_Assistant
   ```

3. **Set Up a Virtual Environment and Install Dependencies**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**

   Create a `.env` file in the root directory with the necessary variables:

   ```env
    OPENAI_API_KEY=openai_api_key
    OPENAI_MODEL=openai_model

    DB_URL=db_url

    LOG_LEVEL=INFO

    APP_NAME=BookBot
    API_PREFIX=/api/v1

    CHATBOT_PROMPT_PATH=prompts/main_chatbot_prompt.txt
    TEXT_TO_SQL_PROMPT_PATH=prompts/text2sql_prompt.txt
    IS_SAFE_SQL_PROMPT_PATH=prompts/is_safe.txt
    PLAN_GENERATION_PROMPT_PATH=prompts/plan_generation_prompt.txt
    EXECUTOR_PROMPT_PATH=prompts/executor_prompt.txt
    TASK_RELEVENCY_PROMPT_PATH=prompts/task_relevency_prompt.txt
    TASK_GENERATION_PROMPT_PATH=prompts/task_generation_prompt.txt

    CHAT_HISTORY_TABLE_NAME=table_name
    CHAT_HISTORY_PRIMARY_KEY_NAME=primary_key_name
   ```

5. **Run the Application**

   ```bash
   uvicorn --reload --host 0.0.0.0 --port 5000 main:app
   ```

## Deployment on AWS Lambda

This repository is deployed on AWS Lambda using the Mangum adapter. The integration with AWS Lambda enables a serverless deployment of the FastAPI application, ensuring scalability and efficient resource utilization. The Lambda deployment allows the chatbot to handle requests seamlessly in a cloud environment.

## Usage

Once the application is up and running, you can interact with the chatbot by sending HTTP POST requests to the `/query` endpoint. Use your favorite REST client (such as Postman) or integrate with the front-end application from one of the other repositories.

The chatbot processes your natural language queries, generates safe and optimized SQL queries, and returns information from the books database.

## Contact

For questions, suggestions, or support, please reach out to the maintainer at [a.elshehawy64@gmail.com](mailto:a.elshehawy64@gmail.com).

---
This repository is maintained as part of a broader system. For a complete overview, please refer to the documentation provided in the other related repositories.
