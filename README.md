# Agente Académico — Instituto Educativo Horizonte

Agente que responde preguntas en lenguaje natural sobre el manual
académico del semestre (calendario, profesores, proyectos, calificaciones)
usando RAG (Retrieval-Augmented Generation).

## Arquitectura
1. El PDF se divide en fragmentos con LangChain (RecursiveCharacterTextSplitter).
2. Cada fragmento se convierte en un embedding con sentence-transformers.
3. Los embeddings se guardan en un índice FAISS.
4. Ante una pregunta, se recuperan los fragmentos más relevantes y se
   envían junto con la pregunta al modelo de lenguaje (Cohere command-a-03-2025).
5. Streamlit expone todo esto como una aplicación de chat, con botones de
   preguntas rápidas y un filtro que rechaza preguntas no académicas.

## Cómo ejecutar el proyecto localmente
1. Clonar el repositorio y entrar a la carpeta.
2. 'python -m venv venv' y activarlo.
3. 'pip install -r requirements.txt'
4. Copiar '.env.example' a '.env' y colocar tu propia clave de Cohere.
5. 'streamlit run app.py'