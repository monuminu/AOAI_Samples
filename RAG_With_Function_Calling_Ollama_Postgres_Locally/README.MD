# RAG with Function Calling Ollama and Postgres Locally

## Overview

This project demonstrates a Retrieval-Augmented Generation (RAG) system with function calling capabilities using Ollama and Postgres, all set up to run locally. The repository includes necessary scripts, Docker configurations, and a demo notebook to get you started.

## Repository Structure

- `.chainlit/`: Directory containing Chainlit configuration files.
- `Dockerfile`: Docker configuration to set up the environment.
- `demo.ipynb`: Jupyter notebook to demonstrate the RAG system.
- `dockerignore.txt`: Docker ignore file to exclude unnecessary files from the Docker build.
- `entrypoint.sh`: Entrypoint script for Docker container.
- `pyproject.toml`: Python project configuration.
- `requirements.txt`: Python dependencies.

## Getting Started

### Prerequisites

- Docker
- Docker Compose
- Python 3.8 or higher
- Jupyter Notebook (optional, for running the demo notebook)

### Installation

1. **Clone the Repository:**
    ```sh
    git clone https://github.com/monuminu/AOAI_Samples.git
    cd AOAI_Samples/RAG_With_Function_Calling_Ollama_Postgres_Locally
    ```

2. **Build and Run Docker Container:**
    ```sh
    docker-compose up --build
    ```

### Running the Demo

Open `demo.ipynb` in Jupyter Notebook to see a step-by-step demonstration of the RAG system with function calling capabilities.

## Configuration

- **Dockerfile:** Defines the Docker image setup.
- **entrypoint.sh:** Script executed when the Docker container starts.
- **pyproject.toml:** Contains project metadata and dependencies.
- **requirements.txt:** Lists Python packages required for the project.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

Special thanks to the contributors and the open-source community.

---

Feel free to explore, modify, and improve the repository. If you encounter any issues or have suggestions, please open an issue or a pull request. Happy coding!
