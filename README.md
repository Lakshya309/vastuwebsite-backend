# Devta Microservice

This is the Python-based FastAPI microservice for the Vastu Spatial Engine. It handles geometry processing, 45 Devta mandala generation, and zone calculation.

## Prerequisites
- Python 3.8+

## How to Start Locally

1. **Navigate to the microservice folder:**
   ```bash
   cd devta_microservice
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment:**
   - **Windows:**
     ```bash
     .venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source .venv/bin/activate
     ```

4. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Start the development server:**
   ```bash
uvicorn main:app --reload --port 5000
   ```

The server should now be running at `http://127.0.0.1:8000`. 
You can view the interactive API documentation at `http://127.0.0.1:8000/docs`.
