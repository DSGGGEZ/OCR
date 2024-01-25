# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Install any needed packages specified in requirements.txt
# (replace 'requirements.txt' with the actual name of your requirements file)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies including libgl1-mesa-glx
# (this package provides libGL.so.1)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . .

# Make port 80 available to the world outside this container
# (adjust the port if needed)
EXPOSE 80

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["uvicorn", "main:app","--host", "0.0.0.0", "--reload"]
