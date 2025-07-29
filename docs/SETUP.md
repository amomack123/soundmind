# SETUP Instructions for SoundMind Project

## Prerequisites
Before you begin, ensure you have the following installed on your machine:
- Python 3.8 or higher
- Docker
- Docker Compose
- Kafka

## Setup Instructions

### 1. Clone the Repository
Clone the SoundMind repository to your local machine:
```
git clone https://github.com/yourusername/soundmind.git
cd soundmind
```

### 2. Create a Virtual Environment
It is recommended to create a virtual environment for managing dependencies:
```
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Dependencies
Install the required Python packages:
```
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file based on the `.env.example` provided:
```
cp .env.example .env
```
Edit the `.env` file to include your specific configurations, such as `MODAL_TOKEN`.

### 5. Set Up Docker Containers
Use Docker Compose to set up Kafka and Zookeeper:
```
docker-compose up -d
```

### 6. Run the Application
You can start the FastAPI application using:
```
uvicorn api.main:app --reload
```

### 7. Testing the Setup
To test the setup, you can run the end-to-end test script:
```
python scripts/e2e_test.py
```

### 8. Additional Documentation
Refer to `docs/BENCHMARKS.md` for optimization metrics and `docs/ARCHITECTURE.md` for diagrams and data flow explanations.

## Conclusion
You are now set up to use the SoundMind project. For any issues, please refer to the documentation or open an issue in the repository.