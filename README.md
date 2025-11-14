# Tutorial de LangChain con OpenAI API

![LangChain](https://img.shields.io/badge/LangChain-Framework-blue)
![OpenAI](https://img.shields.io/badge/OpenAI-API-green)
![Python](https://img.shields.io/badge/Python-3.10%2B-yellow)
![License](https://img.shields.io/badge/License-MIT-red)

##  Descripción del Proyecto

Este repositorio contiene un tutorial completo sobre **LangChain**, una librería avanzada diseñada para construir aplicaciones impulsadas por modelos de lenguaje de gran escala (LLMs). El proyecto integra la API de OpenAI y demuestra cómo crear flujos de trabajo inteligentes mediante cadenas de prompts, gestión de memoria conversacional y composición de respuestas estructuradas.


## Contenido del Repositorio

Este repositorio incluye los siguientes archivos y notebooks educativos:

| Archivo | Descripción |
|---------|-------------|
| **setup_hello_ai.ipynb** | Notebook de configuración inicial para APIs de IA |
| **Guia3_IntroAPIsAI_Notebook.ipynb** | Introducción práctica a APIs de IA con OpenAI |
| **Guia4_Introduccion_LangChain_OpenAI.ipynb** | Tutorial principal de LangChain con ejemplos completos |
| **Guia5_HuggingFace_Intro.ipynb** | Introducción al ecosistema Hugging Face y transformers |
| **.env** | Archivo de variables de entorno (crear localmente)


### Descripción de los Notebooks

#### 1. `setup_hello_ai.ipynb`
Notebook inicial para instalar dependencias básicas:
- Instalación de `openai` y `python-dotenv`
- Configuración del entorno de trabajo

#### 2. `Guia3_IntroAPIsAI_Notebook.ipynb`
Introducción completa a las APIs de modelos de lenguaje:
- Configuración de credenciales con archivos `.env`
- Parámetros clave: `temperature`, `max_tokens`, `top_p`
- Primeras consultas a OpenAI API
- Respuestas estructuradas en JSON
- Buenas prácticas y solución de problemas

#### 3. `Guia4_Introduccion_LangChain_OpenAI.ipynb` 
Tutorial completo de LangChain con tres ejemplos prácticos:
- **Ejemplo 1:** Prompts simples con LCEL (LangChain Expression Language)
- **Ejemplo 2:** Cadenas secuenciales (multi-step chains)
- **Ejemplo 3:** Memoria conversacional para chatbots educativos
- Integración con OpenAI API


#### 4. `Guia5_HuggingFace_Intro.ipynb`
Introducción al ecosistema Hugging Face:
- Modelos de transformers (BERT, GPT-2, DistilBERT)
- Pipelines para clasificación, generación y embeddings
- Uso de datasets públicos
- Ejecución local vs remota (Inference API)
- Tokenización y análisis de sentimientos

---

## Arquitectura del Proyecto

### Estructura del Proyecto

```
TallerOpenIA/
│
├── .env                                    # Variables de entorno (NO subir a Git)
├── .gitignore                              # Archivos excluidos del control de versiones
├── README.md                               # Este archivo
├── setup_hello_ai.ipynb                    # Setup inicial
├── Guia3_IntroAPIsAI_Notebook.ipynb       # Introducción a APIs de IA
├── Guia4_Introduccion_LangChain_OpenAI.ipynb  # Tutorial principal de LangChain
└── Guia5_HuggingFace_Intro.ipynb          # Introducción a Hugging Face
```

### Componentes Principales

```
┌─────────────────────────────────────────────────────────────┐
│                    APLICACIÓN LANGCHAIN                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────┐      ┌─────────────────┐               │
│  │  Prompt        │──────│  LLM Chain      │               │
│  │  Templates     │      │  (LCEL)         │               │
│  └────────────────┘      └─────────────────┘               │
│                                 │                            │
│                                 ▼                            │
│  ┌────────────────┐      ┌─────────────────┐               │
│  │  Memoria       │◄─────│  ChatOpenAI     │               │
│  │  Conversacional│      │  (gpt-4o-mini)  │               │
│  └────────────────┘      └─────────────────┘               │
│                                 │                            │
│                                 ▼                            │
│                          ┌─────────────────┐                │
│                          │  Output Parser  │                │
│                          │  (Structured)   │                │
│                          └─────────────────┘                │
└─────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │   OpenAI API    │
                        │   (Remoto)      │
                        └─────────────────┘
```
---

## Instalación y Configuración

### Requisitos Previos

- **Python 3.10 o superior**
- **Cuenta de OpenAI** con clave API activa
- **Editor recomendado:** Visual Studio Code, Jupyter Notebook o Google Colab

### Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/LIZVALMU/TallerOpenIA.git
cd TallerOpenIA
```

### Paso 2: Configurar Variables de Entorno

1. Abrir VS Code en esta carpeta del proyecto
2. Presione Ctrl+Shift+P (Windows/Linux) 
3. Escriba Python: Create Environment y presione Enter
4. Seleccione Venv
5. Elija el intérprete de Python (Python 3.10+)
6. Espere a que VS Code cree el entorno automáticamente
7. Acepte activar el entorno en el proyecto


### Paso 3 Configurar la clave API
1. Crea un archivo `.env` en la raíz del proyecto con el siguiente contenido:


```bash
OPENAI_API_KEY=sk-your-actual-api-key-here
```

IMPORTANTE: No compartir ni subir este archivo a repositorios públicos

---


##  Ejemplos de Código en Funcionamiento
### 1. Setup Inicial (`setup_hello_ai.ipynb`)

Este notebook instala las dependencias básicas necesarias:

```python
%pip install openai python-dotenv
```

**Salida esperada:**
```
Collecting openai
  Downloading openai-1.12.0-py3-none-any.whl (226 kB)
Collecting python-dotenv
  Downloading python_dotenv-1.0.0-py3-none-any.whl (19 kB)
Successfully installed openai-1.12.0 python-dotenv-1.0.0
```

---

### 2. Primera Consulta a OpenAI API (`Guia3_IntroAPIsAI_Notebook.ipynb`)

**Código de inicialización:**
```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print("Cliente inicializado. Modelo listo para consultas.")
```

**Salida:**
```
Cliente inicializado. Modelo listo para consultas.
```

**Primera consulta con control de temperatura:**
```python
prompt = "Explica en dos frases qué es la inteligencia artificial en la educación."
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7
)
print(response.choices[0].message.content)
```

**Salida esperada:**
```
La inteligencia artificial en la educación se refiere al uso de tecnologías avanzadas para personalizar el aprendizaje, automatizar tareas administrativas y proporcionar retroalimentación instantánea a los estudiantes. Estas herramientas pueden adaptar el contenido educativo según las necesidades individuales de cada alumno, mejorando la eficiencia y efectividad del proceso de enseñanza-aprendizaje.
```

**Respuesta estructurada en JSON:**
```python
import json

query = "¿Qué es aprendizaje supervisado?"
schema_instruction = (
    "Responde en formato JSON con las claves: operation, input, output. "
    "operation debe ser 'explanation'; input debe repetir la pregunta; "
    "output la explicación clara y breve."
)
response_json = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": schema_instruction},
        {"role": "user", "content": query}
    ],
    temperature=0.3,
    max_tokens=300
)
text = response_json.choices[0].message.content
print(text)
```

**Salida esperada:**
```json
{
  "operation": "explanation",
  "input": "¿Qué es aprendizaje supervisado?",
  "output": "El aprendizaje supervisado es un tipo de machine learning donde el 
  modelo aprende de datos etiquetados, es decir, ejemplos que incluyen tanto 
  entradas como salidas esperadas. El objetivo es que el modelo aprenda a 
  predecir la salida correcta para nuevas entradas no vistas previamente."
}
```

---

### 3. LangChain - Prompt Simple (`Guia4_Introduccion_LangChain_OpenAI.ipynb`)

**Inicialización del cliente LangChain:**
```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
print("Cliente LangChain con OpenAI inicializado correctamente.")
```

**Salida:**
```
Cliente LangChain con OpenAI inicializado correctamente.
```

**Cadena simple con LCEL:**
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

prompt = ChatPromptTemplate.from_template(
    "Explica en dos frases el concepto de {tema}."
)

chain = prompt | llm | StrOutputParser()

result = chain.invoke({"tema": "aprendizaje automático"})
print(result)
```

---

### 4. LangChain - Cadenas Secuenciales

**Encadenamiento de múltiples pasos:**
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
to_str = StrOutputParser()

# Paso 1: Explicar el concepto
primer_prompt = ChatPromptTemplate.from_template(
    "Explica brevemente el concepto de {tema}."
)
primer_paso = primer_prompt | llm | to_str

# Paso 2: Proponer aplicación educativa
segundo_prompt = ChatPromptTemplate.from_template(
    "Propón una aplicación educativa del siguiente concepto: {concepto}."
)
segundo_paso = segundo_prompt | llm | to_str

# Encadenar
cadena_secuencial = {"concepto": primer_paso} | segundo_paso

resultado = cadena_secuencial.invoke({"tema": "realidad aumentada"})
print(resultado)
```
---

### 5. LangChain - Memoria Conversacional

**Chatbot con contexto:**
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
to_str = StrOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente educativo claro y conciso."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

chain = prompt | llm | to_str
history = []

def chat(user_text: str) -> str:
    global history
    answer = chain.invoke({"input": user_text, "chat_history": history})
    history += [HumanMessage(content=user_text), AIMessage(content=answer)]
    return answer
```

---

### 6. Hugging Face - Clasificación de Sentimientos (`Guia5_HuggingFace_Intro.ipynb`)

**Pipeline básico:**
```python
from transformers import pipeline

# Clasificación de sentimientos
clf = pipeline("sentiment-analysis")
result = clf("Este curso de IA en el aula me parece excelente.")
print(result)
```

**Salida esperada:**
```
[{'label': 'POSITIVE', 'score': 0.9998}]
```

**Análisis detallado con AutoModel:**
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_id = "distilbert-base-uncased-finetuned-sst-2-english"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

inputs = tok("I love practical AI courses.", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
pred = torch.softmax(logits, dim=-1).tolist()[0]
print({"NEGATIVE": pred[0], "POSITIVE": pred[1]})
```

**Salida esperada:**
```
{'NEGATIVE': 0.0008, 'POSITIVE': 0.9992}
```

---

### 7. Hugging Face - Embeddings (Representaciones Vectoriales)

**Generación de embeddings:**
```python
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

emb_model_id = "sentence-transformers/all-MiniLM-L6-v2"
emb_tok = AutoTokenizer.from_pretrained(emb_model_id)
emb_model = AutoModel.from_pretrained(emb_model_id)

def embed(texts):
    batch = emb_tok(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        out = emb_model(**batch)
    tokens = out.last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1)
    masked = tokens * mask
    sent_emb = masked.sum(dim=1) / mask.sum(dim=1)
    return F.normalize(sent_emb, p=2, dim=1)

e = embed([
    "inteligencia artificial en educación", 
    "clase de programación", 
    "oxigenación en hidroeléctricas"
])
print("Shape de embeddings:", e.shape)

# Similitud coseno entre primera y segunda frase
sim = (e[0] @ e[1]).item()
print(f"Similitud coseno (frase 1 vs 2): {round(sim, 4)}")

# Similitud entre primera y tercera (no relacionadas)
sim_no_rel = (e[0] @ e[2]).item()
print(f"Similitud coseno (frase 1 vs 3): {round(sim_no_rel, 4)}")
```

**Salida esperada:**
```
Shape de embeddings: torch.Size([3, 384])
Similitud coseno (frase 1 vs 2): 0.4823
Similitud coseno (frase 1 vs 3): 0.1247
```
---

##  Solución de Problemas Comunes

### Error: `openai.AuthenticationError`

**Causa:** Clave API inválida o no cargada.

**Solución:**
```bash
# Verificar que el archivo .env existe y contiene la clave
type .env  # En Windows

# Reiniciar el kernel de Jupyter si es necesario
```

### Error: `Rate limit exceeded`

**Causa:** Excediste el límite de solicitudes por minuto.

**Solución:**
```python
import time
time.sleep(2)  # Esperar 2 segundos entre llamadas
```

### Error: `model_not_found`

**Causa:** El modelo especificado no existe o no tienes acceso.

**Solución:**
```python
# Usar un modelo disponible en tu plan
llm = ChatOpenAI(model="gpt-3.5-turbo")  # Modelo base
```

##  Buenas Prácticas

### Hacer

- Usar prompts **claros y específicos**
- Controlar `temperature` según la tarea
- Guardar logs de conversaciones para análisis
- Limitar `max_tokens` para controlar costos
- Documentar las cadenas para reutilización
- Validar salidas antes de usarlas en producción

###  Evitar

- Exponer claves API en el código
- Usar `temperature` alto para tareas de precisión
- Omitir manejo de errores en llamadas a la API
- Procesar datos personales sin consentimiento

---
## Autora

**Autora:** Alison Geraldine Valderrama Munar  
**Curso:** AREP - Arquitecturas Empresariales  
**Universidad:** Escuela Colombiana de Ingeniería Julio Garavito

---

## Licencia
Este proyecto fue desarrollado como parte del curso de Arquitecturas Empresariales de la Escuela Colombiana de Ingeniería Julio Garavito.
