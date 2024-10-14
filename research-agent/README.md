 # Multi-Agent Research Task Runner

This project is designed to execute research tasks using a multi-agent system. It leverages the `ChiefEditorAgent` to process tasks defined in a `task.json` file.

## Prerequisites

- Python 3.7+
- `dotenv` for environment variable management
- `asyncio` for asynchronous operations

## Setup

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install dependencies:**

   Ensure you have `dotenv` and any other required packages installed:

   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables:**

   Create a `.env` file in the root directory and set your environment variables. If you have a `LANGCHAIN_API_KEY`, it will enable LangSmith tracing.

   ```
   LANGCHAIN_API_KEY=your_api_key_here
   ```

4. **Task Configuration:**

   Ensure there is a `task.json` file in the same directory as `main.py`. This file should contain the necessary task information.

## Usage

Run the main script to execute the research task:

## Functions

- **`open_task()`**: Loads the task configuration from `task.json`.
- **`run_research_task(query, websocket=None, stream_output=None, tone=None, headers=None)`**: Executes a research task with the given parameters.
- **`main()`**: The main entry point for running the research task.

## Notes

- The `ChiefEditorAgent` is responsible for processing the task and generating a research report.
- If a WebSocket and stream output function are provided, the research report will be streamed in real-time.

## License

This project is licensed under the MIT License.