#start with a lightweight python image
#this gives our container a base python runtime environment to run our code
FROM python:3.14-slim

#set the working directory in the container
WORKDIR /app

#copy the dependency list before copying application code
COPY requirements.txt .

#avoid retaining pip's download cache in the image
RUN pip install --no-cache-dir -r requirements.txt

#copy the application code and saved model into /app
COPY . .

#Document the port used by FastAPI application
EXPOSE 8000

#start the FastAPI server when the container launches
#0.0.0.0 allows connections through the container's external network interface
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]




