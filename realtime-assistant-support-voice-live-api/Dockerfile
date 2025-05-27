FROM mcr.microsoft.com/devcontainers/python:3.11

ENV PORT=8080
# Set the working directory
WORKDIR /app
# Copy your application's requirements and install them
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt
# Copy your application code into the container
COPY . /app/
EXPOSE 8080

# Command to run the application
CMD ["python", "-m", "chainlit", "run", "app.py", "-h", "--host", "0.0.0.0", "--port", "8080"]