#  This is the Docker file

# Use an official Python runtime as a parent image
FROM python:3
# Set the working directory to /app
WORKDIR /app
# Copy requirements.txt into the container
COPY ./requirements.txt /app/requirements.txt
# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt
# Copy the current work directory /app
COPY . /app
# Make port 5002 visible
EXPOSE 5002
# Define environment variable
ENV NAME World
# Run app.py when the container launches
CMD ["python", "app_dir/form_app.py"]
