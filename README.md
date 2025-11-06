# Chatbot PagoPa


# Run Docker Compose
To run ollama server and neo4j with docker you can use the docker compose file
``` sh
cd docker/
docker compose up -d
```

# Virtual Env configuration
Set your python virtual env
``` sh
python3 -m venv chatbot
source chatbot/bin/activate
```

# Install Requirements
To install the dependencies use this command:
``` sh
pip install -r requirements.txt
``` 

# Run App
To run the app first of all you need to popolate the database with:
``` sh
cd src/
python3 init.py
```

Then you can run the chatbot with the main script
``` sh
python3 main.py
```
