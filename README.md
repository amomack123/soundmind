# 🧠 SoundMind - Production Scaffold

## Overview
SoundMind is a comprehensive audio processing framework designed to handle various audio tasks, including separation, classification, enhancement, and transcription. This project utilizes a modular architecture, allowing for easy integration and scalability of different audio processing jobs.

## Project Structure
The project is organized into several key directories:

- **agents/**: Contains the core logic for managing audio processing tasks, including a DAG planner and task routing.
- **api/**: Provides an interface for interacting with the audio processing system, including a FastAPI application and a command-line interface.
- **modal_jobs/**: Houses the various audio processing jobs, each encapsulated in its own worker with Docker support.
- **pubsub/**: Implements a publish-subscribe model for job messaging using Kafka.
- **shared/**: Contains shared utilities, schemas, and constants used across the project.
- **scripts/**: Includes scripts for testing and logging performance metrics.
- **infra/**: Contains infrastructure setup files, including Docker Compose configurations for Kafka and Zookeeper.
- **docs/**: Provides documentation for setup, benchmarks, and architecture.

## Getting Started
To get started with SoundMind, follow the instructions in the `docs/SETUP.md` file for setting up the necessary environment and dependencies.

## Usage
Once the setup is complete, you can use the API to submit audio jobs and monitor their processing through the CLI or directly via the FastAPI interface.

## Contributing
Contributions are welcome! Please refer to the `README.md` for guidelines on how to contribute to the project.

## License
This project is licensed under the MIT License - see the LICENSE file for details.