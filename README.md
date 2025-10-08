[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]


![https://welearn-demo.learningplanetinstitute.org](https://welearn-demo.learningplanetinstitute.org/assets/logo-fdaefaa7.png)

## About Us

This repository is maintained by Learning Planet Institute, a non-governmental organization dedicated to The Learning Planet Institute is a global NGO dedicated to transforming education through innovative, collaborative, and inclusive approaches. It fosters the development of sustainable learning ecosystems, promotes lifelong learning, and empowers individuals and communities to actively contribute to solving global challenges. The institute connects diverse stakeholders to create a more equitable and impactful educational future.

## WeLearn API


This project is a web application built with **FastAPI** and **Poetry** for dependency management. It is designed to provide an interface for clients to search for **SDG-classified documents** and leverage **chat capabilities** enhanced by these SDG documents.

The application offers powerful **search** and **chat** endpoints that enable users to:

- **Search for SDG-classified documents**: Clients can search through documents tagged with specific **Sustainable Development Goals (SDGs)** using advanced similarity-based search techniques.
- **Chat functionality**: The chat interface is powered by a **Large Language Model (LLM)**, which is enriched with SDG-specific information to deliver more accurate and context-aware responses.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Requirements](#requirements)
- [Docker Setup](#docker-setup)
- [API Dependency](#api-dependency)
- [Project Structure](#project-structure)
- [License](#license)
- [Contact](#contact)

## Introduction

This project is built with **FastAPI**, **Poetry**, and **Docker Compose**. It creates a **REST API** that allows the client to interact with various services, including:

- **Qdrant Database**: Stores vector embeddings and allows for similarity-based searches, enabling efficient document retrieval.
- **Postgres Database**: A relational database used for storing structured data, such as metadata or user information.
- **LLM Service**: Enhances the client’s experience by generating responses based on the data stored in both Qdrant and Postgres databases.

The application exposes several API endpoints that allow clients to:
- **Search for documents** in the Qdrant database using similarity search.
- **Enhance LLM responses** by utilizing the retrieved documents to provide more relevant, context-aware, and accurate answers.

By combining document search and LLM capabilities, the app ensures intelligent, data-driven interactions, making it easier for users to get the most relevant information.
While it is open-source, **please note that this project is not currently open to external contributions**.

## Features

## API Endpoints

The project exposes several REST API endpoints that allow clients to interact with various features. Below is a list of the main endpoints available in the API:

### Search Endpoints

- **GET** `/api/v1/search/collections`  
  **Description**: Retrieves the corpus or collections of documents available for searching.

- **POST** `/api/v1/search/collections/{collection_query}`  
  **Description**: Performs a search query on a specific collection to retrieve relevant items.  

- **POST** `/api/v1/search/by_slices`  
  **Description**: Allows searching for documents across all slices (segments) by language.

- **POST** `/api/v1/search/by_document`  
  **Description**: Performs a search on all documents, helping to locate content based on specific document data.

### Q&A Operations (OpenAI Integration)

- **POST** `/api/v1/qna/reformulate/query`  
  **Description**: Reformulates a given query to improve clarity or context for the LLM (Large Language Model).

- **POST** `/api/v1/qna/reformulate/questions`  
  **Description**: Reformulates a set of questions to enhance understanding or to generate more context-specific queries.

- **POST** `/api/v1/qna/chat/rephrase`  
  **Description**: Rephrases a chat message to improve its quality or alter its expression while maintaining the original meaning.

- **POST** `/api/v1/qna/chat/rephrase_stream`  
  **Description**: Provides a streaming response of rephrased content during a chat, allowing for dynamic rephrasing.

- **POST** `/api/v1/qna/chat/answer`  
  **Description**: Provides an answer to a question from the chat interface, utilizing the LLM and relevant data.

- **POST** `/api/v1/qna/stream`  
  **Description**: Streams responses from the Q&A service, providing real-time answers to the user’s questions.

- **POST** `/api/v1/qna/chat/agent`  
  **Description**: Interact in a conversation from the chat interface, using the LLM, and fetching relevant data from WeLearn only if needed.

---

### Example Usage of Search Endpoints

Here’s an example of how you can use the **Search** endpoints:

#### Search Items in a Collection
```bash
curl -X 'POST' \
  'https://API_URL/api/v1/search/collections/COLLECTION_NAME?query=QUERY_TO_SEARCH&nb_results=10' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "sdg_filter": [
    1,
    2,
    3
  ]
}'
```

## Requirements

To run this project locally, you'll need the following tools:

- **Python** (version >= 3.10)
- **Poetry** (version >= 2.1): You can install Poetry by following the instructions [here](https://python-poetry.org/docs/#installation).
- **Docker** (optional, for Docker setup)
- **Docker Compose** (optional, for Docker setup)

### Install Poetry:

To install Poetry, run the following command:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Once installed, you can verify Poetry is installed correctly by running:

```bash
poetry --version
```

## Installation

To install and run this project locally, follow these steps:

1. Clone the repository:
```bash
    git clone https://github.com/your-username/project-name.git
```
2. Navigate into the project directory:
```bash
cd WeLearn-api
```
3. Install the dependencies using Poetry:
```bash
poetry install
```
4. Copy and configure the environment variables:
```bash
cp .env.example .env
```
5. Update the .env file with the appropriate values for your local setup.
6. (Optional) If you prefer to run the app with Docker Compose, see Docker Setup.

## Usage

Once you have the app set up locally, you can use the application by navigating to the appropriate URL in your browser.

Run the FastAPI app
To run the FastAPI app locally (without Docker), use the following command:

```bash
poetry run uvicorn app.main:app --reload
```

The app will be available at http://localhost:8000/.

Run Common Development Tasks with Makefile
We’ve included a Makefile for common tasks to streamline the setup and development process. For example:

Run Tests:

```bash
make test
```

Run App:

```bash
make run-poetry
```

For more details on available tasks, check the Makefile or run make to see the list of available commands.

### API Dependency
This project relies on an external REST API for certain features. Some functionality of the application may not work properly if the API is down or unreachable.

You can check the status and setup instructions for the API in the API repository.

### Docker Setup
If you prefer to run the application using Docker, follow these steps:

Ensure Docker and Docker Compose are installed on your system.

Run the following command to start the application and its dependencies:

```bash
docker-compose up
```

The app should now be available at http://localhost:8000/.

To stop the app, use:

```bash
docker-compose down
```

## Project Structure
Here’s an overview of the project directory structure:

- **`app/`**: Contains the FastAPI application code.
- **`app/main.py`**: The entry point of the FastAPI app.
- **`app/api/`**: Contains the different API routes.
- **`app/models/`**: Contains the Pydantic models and database models.
- **`app/services/`**: Contains business logic or services.
- **`app/utils/`**: Contains utility functions used across the app.
- **`docker-compose.yml`**: The configuration for Docker Compose.
- **`Dockerfile`**: The Dockerfile for building the application container.
- **`tests/`**: Contains unit and integration tests for the application.
- **`.env.example`**: Example environment variable file. Copy this to .env and configure it with your settings.
- **`Makefile`**: Contains commonly used commands like make run or make test.

## License

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

## Contact
If you have any questions or need further information, feel free to reach out at:

Email: welearn@learningplanetinstitute.org
