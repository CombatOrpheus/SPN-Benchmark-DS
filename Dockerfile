# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependency file first to leverage Docker cache
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
# Using --no-cache-dir reduces the image size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code
COPY . .

# The original CMD was ["pip", "list"], which is not very useful.
# A better default would be to open a bash shell to allow the user
# to run any script they want.
CMD ["/bin/bash"]
