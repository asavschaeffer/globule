# Globule: Your Personal Thought-Processor

Welcome to Globule, a command-line tool for capturing, understanding, and retrieving your thoughts. It's like a second brain, but on your local machine.

## Key Features

*   **Instant Capture:** Jot down ideas, notes, and reminders with a single command.
*   **Semantic Search:** Find what you're looking for based on meaning, not just keywords.
*   **Daily Summaries:** Get a daily report of your thoughts and activities.
*   **Local First:** Everything is stored on your machine, so your data is always private.

## Getting Started

Getting started with Globule is easy. Just follow these three steps:

1.  **Install Dependencies:**

    ```bash
    poetry install
    ```

2.  **Configure Globule:**

    ```bash
    poetry run globule config
    ```

3.  **Start Ollama:**

    ```bash
    # In a separate terminal
    ollama serve
    ```

## Usage

Here are a few examples of how to use Globule:

*   **Add a thought:**

    ```bash
    poetry run globule add "I had a great idea for a new project today."
    ```

*   **Search for a thought:**

    ```bash
    poetry run globule search "new project idea"
    ```

*   **Get a daily summary:**

    ```bash
    poetry run globule report
    ```

## How It Works

Globule uses a combination of local AI models to understand and process your thoughts:

*   **mxbai-embed-large:** Creates semantic embeddings for each thought.
*   **llama3.2:3b:** Extracts entities, categories, and sentiment.

This allows Globule to perform intelligent searches and generate insightful summaries.

## Roadmap

Here are some of the features we're planning to add to Globule:

*   **Web Interface:** A simple web UI for viewing and managing your thoughts.
*   **Mobile App:** A mobile app for capturing thoughts on the go.
*   **Cloud Sync:** The ability to sync your thoughts across multiple devices.

## Contributing

We welcome contributions from the community. If you'd like to contribute, please fork the repository and submit a pull request.

## License

Globule is licensed under the MIT License. See the `LICENSE` file for more information.