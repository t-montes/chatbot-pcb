# `Chatbot Financiero` using GenAI

## Configuration

Set the `global.json` file with the required attributes, following the [`template`](./config/global-template.json)

Copy the JSON key file from the GCP project into the config folder, and reference it from the `global.json` "service_account_key" attribute.

## Usage

The general commands before using it, are:

```shell
$ git clone https://github.com/t-montes/chatbot-financiero
$ cd chatbot-financiero
```

### Development and Testing

```shell
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
$ streamlit run app.py --server.enableCORS=false --server.enableXsrfProtection=false --server.port=8080
```

### Production

```shell
$ docker build -t chatbot-financiero-genai .
$ docker run -p 80:8080 chatbot-financiero-genai
```

In both cases, a web browser will open with the chatbot interface, at port 8080
