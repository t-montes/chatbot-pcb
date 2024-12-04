# ************************* Chatbot Financiero GenAI *************************
import google.cloud.bigquery as BigQuery
from google.oauth2 import service_account
from langchain_google_vertexai import VertexAI
from langchain.prompts.prompt import PromptTemplate
from langchain_community.document_loaders import BigQueryLoader
from langchain.schema import format_document
from unidecode import unidecode
from datetime import datetime, date
import random
import utils
import json
import yaml
import time
import re

# ************************ Config and Init Variables ************************
with open("config/global.json", "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

with open("agent-config/examples.txt", "r", encoding="utf-8") as f:
    EXAMPLES = []
    idx = 0
    for line in f:
        if line.strip() == "":
            idx += 1
        else:
            if len(EXAMPLES) <= idx:
                EXAMPLES.append("")
            EXAMPLES[idx] += line

MODELS = """\
  * predicción de ventas: sales_model
  * predicción de costos: costs_model\
"""

ADDITIONAL_INFO = """\
  * Para los datos predictivos, se utilizó una estimación por medio de un modelo ARIMA (Modelo autorregresivo integrado de media móvil) entrenado con los datos históricos
  * Para las consultas descriptivas, se genera y realiza una consulta SQL sobre la base de datos de la empresa
  * Las preguntas de conversación pueden ser simples preguntas como "Hola", "Describe los datos que tienes", "Cómo puedo dividir las ventas por ubicación?"
  * O también pueden ser preguntas relacionadas con respuestas anteriores, como: 
    * "Por qué la predicción de ventas es tan baja?" (siempre y cuando la pregunta anterior haya sido una predicción de ventas)\
    * "Compara estos resultados" (considerando las preguntas/respuestas dadas anteriormente)\
"""

PROJECT_ID = CONFIG["global"]["project_id"]
DATASET_ID = CONFIG["global"]["dataset_id"]
LOCATION_ID = CONFIG["global"]["location_id"]
VERTEX_AI_MODEL = CONFIG["global"]["vertex_ai_model"]

CREDENTIALS = service_account.Credentials.from_service_account_file(CONFIG["global"]["service_account_key"])
CREDENTIALS = CREDENTIALS.with_scopes([scope.strip() for scope in CONFIG["global"]["authentication_scope"].split(',')])

BIGQUERY_CLIENT = BigQuery.Client(credentials=CREDENTIALS)

SCHEMAS_QUERY = f"""
SELECT table_catalog, table_schema, table_name, column_name, data_type
FROM `{PROJECT_ID}.{DATASET_ID}.INFORMATION_SCHEMA.COLUMN_FIELD_PATHS`;
"""
BQLOADER = BigQueryLoader(SCHEMAS_QUERY, page_content_columns=["table_catalog", "table_schema", "table_name", "column_name", "data_type"], credentials=CREDENTIALS)
SCHEMAS_DOCS = BQLOADER.load()

JSON_PATTERN = r"```json(.*?)```" 

with open("./agent-config/agent.yaml", "r", encoding="utf-8") as file:
    prompts = yaml.safe_load(file)["prompts"]

# ************************ Functions ************************
def extract_json(response, step):
    try:
        match = re.search(JSON_PATTERN, response, re.DOTALL)
        if match:
            return json.loads(match.group(1).strip())
        else:
            return json.loads(response)
    except: 
        raise ValueError(f"JSON field not found. Step: {step}")

def run_sql(clean_query):
    try:
        df = BIGQUERY_CLIENT.query(clean_query).result().to_dataframe()
        sql_error = ""
    except Exception as e:
        df = []
        sql_error = e
    return (sql_error, df)

def get_prompt(name):
    prompt_info = prompts.get(name)
    if not prompt_info:
        raise ValueError(f"Prompt '{name}' not found in the YAML file.")
    
    template = PromptTemplate(
        input_variables=prompt_info["variables"],
        template=prompt_info["template"]
    )
    return template

# ************************ Classes ************************
class LLM():
    def __init__(self, template_name):
        self.llm = VertexAI(project=PROJECT_ID, location=LOCATION_ID, credentials=CREDENTIALS, model_name=VERTEX_AI_MODEL, max_output_tokens=8192, temperature=0)
        self.template = get_prompt(template_name)
    
    def __call__(self, **kwargs):
        prompt = self.template.format(**kwargs)
        print(f"Prompt:\n{prompt}")
        response = self.llm.invoke(prompt)
        print(f"Response:\n{response}")
        return response

class Agent():
    history = []
    def __init__(self, history_length=3):
        self.history_length = history_length
        
        self.sql_generate = LLM("sql_generate")
        self.nl_response = LLM("nl_response")
        self.q_type = LLM("q_type")
        self.talk = LLM("talk")
        self.forecast_q_generate = LLM("forecast_q_generate")
        self.rephrase_q = LLM("rephrase_q")
        self.filter_ctx = LLM("filter_ctx")
        self.is_feasible = LLM("is_feasible")

    def get_history(self):
        context = []
        i,skipthis=0,False
        for message in self.history[-2::-1]:
            if i//2 >= self.history_length: break
            if skipthis:
                skipthis=False
                continue
            if message["status"].startswith("success"): 
                i+=1
                if message['role'] == 'human': 
                    msg = f"Pregunta: {message['content']}"
                    context.append(msg)
                else:
                    msg = f"Respuesta: {message['content']}"
                    msg += f"\nDataframe: {message['dataframe']}" if 'dataframe' in message else ""
                    msg += f"\nSQL Query: {message['sql_query']}" if 'sql_query' in message else ""
                    context.append(msg + '\n')
            else: skipthis=True
        return '\n'.join(context[::-1])
    
    def get_schema(self, schemas_docs):
        return "\n\n".join(
            format_document(doc, PromptTemplate.from_template("{page_content}"))
            for doc in schemas_docs
        )
    
    def get_examples(self, examples):
        return '\n'.join(examples)

    def __call__(self, prompt, callback=lambda*x:x):
        """returns: response, dataframe, sql_query"""
        # -------------------- Context Variables --------------------
        history = self.get_history()
        schema = self.get_schema(SCHEMAS_DOCS)
        examples = self.get_examples(EXAMPLES)
        onlyq_examples = [ex.split("\n")[0] for ex in EXAMPLES]
        today_date = datetime.now()
        today = datetime.now().strftime("%d/%m/%Y")

        # ----------------------- Agent Steps -----------------------
        # 1. Determine the query type
        q_type_r = self.q_type(
            prompt=prompt,
            date=today,
        )
        q_type_j = extract_json(q_type_r, "q_type")
        q_type = q_type_j['query_type']

        match q_type:
            case "description":
                context = f"* **Esquema de tablas en BigQuery:**\n{schema}"
            case "prediction":
                context = f"* **Modelos de forecast disponibles:**\n{MODELS}"
            case "talk":
                context = f"* **Información adicional de la empresa y datos:**\n{ADDITIONAL_INFO}"
            case _:
                raise ValueError(f"Query type '{q_type}' not supported.")
        
        # 2. Check if the query is feasible
        is_feasible_r = self.is_feasible(
            prompt=prompt,
            context=context,
            history=history,
            q_type=q_type,
            date=today,
            models=MODELS,
        )
        is_feasible_j = extract_json(is_feasible_r, "is_feasible")
        reason_feasible = is_feasible_j['reason']
        is_feasible = is_feasible_j['feasible']

        if not is_feasible:
            ...

        # 3. Rephrase the question, considering the history
        new_prompt = self.rephrase_q(
            prompt=prompt,
            context=context,
            history=history,
            date=today,
            reasoning=reason_feasible,
        )

        if q_type == "prediction":
            # 4.1. Generate Query (prediction)
            forecast_q_generate_r = self.forecast_q_generate(
                prompt=new_prompt,
                date=today,
                models=MODELS,
            )
            forecast_q_generate_j = extract_json(forecast_q_generate_r, "forecast_q_generate")
            fc_model = forecast_q_generate_j['model']

            fc_up_date = forecast_q_generate_j['up_date']
            fc_up_date = datetime.strptime(fc_up_date, '%Y-%m-%d').date()
            if fc_up_date > date(2026, 12, 31): raise ValueError("Únicamente se pueden hacer predicciones hasta el 31 de diciembre de 2026.")
            fc_steps = int(((fc_up_date.year - today_date.year) * 12 + fc_up_date.month - today_date.month + (fc_up_date.day - today_date.day) / 30)//1)+1
            if fc_steps < 1: raise ValueError("La fecha de predicción debe ser mayor a la fecha actual.")

            forecast = utils.predict(fc_model, fc_steps)

            # 5.1. Analyze DF in Natural Language (prediction)
            nl_response = self.nl_response(
                prompt=new_prompt,
                date=today,
                dataframe=forecast,
                query=f"call forecast model: {fc_model}(months_in_future={fc_steps})",
            )
            nl_response = re.sub(r'[$€£¥₹]', r'\\$', nl_response)
            return nl_response, True, forecast, None

        if q_type == "description":
            # 4.2. Generate SQL Query (description)
            sql_generate_r = self.sql_generate(
                prompt=new_prompt,
                examples=examples,
                schema=schema,
                date=today,
            )
            clean_query = sql_generate_r.replace("```sql", "").replace("```", "")
            clean_query = unidecode(clean_query)

            sql_error, df = run_sql(clean_query)

            if not sql_error:
                # 5.2. Analyze DF in Natural Language (description)
                nl_response = self.nl_response(
                    prompt=new_prompt,
                    date=today,
                    dataframe=df,
                    query=clean_query,
                )
                nl_response = re.sub(r'[$€£¥₹]', r'\\$', nl_response)
                return nl_response, True, df, clean_query
            else:
                return sql_error, False, None, None

        if q_type == "talk":
            # 4.2. Answer (talk)
            talk = self.talk(
                prompt=new_prompt,
                additional_info=ADDITIONAL_INFO,
                schema=schema,
                date=today,
            ) # direct response
            talk = re.sub(r'[$€£¥₹]', r'\\$', talk)
            return talk, True, None, None
