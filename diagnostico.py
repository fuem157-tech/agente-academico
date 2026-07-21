"""
Script de diagnóstico: NO reemplaza a app.py, es solo para investigar
por qué ciertas preguntas fallan. Ejecútalo con:

    python diagnostico.py

Debe estar en la misma carpeta que app.py y los 6 PDFs, con el venv activo.
"""
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_cohere import ChatCohere
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

load_dotenv()

DOCUMENTOS_PDF = [
    "manual_academico_instituto_horizonte.pdf",
    "reglamento_del_estudiante.pdf",
    "politica_reembolso_matriculas.pdf",
    "faq_cursos_certificados.pdf",
    "guia_uso_plataforma.pdf",
    "programa_becas_afiliados.pdf",
]

print("=" * 80)
print("PASO 1: Cargando y dividiendo los documentos...")
print("=" * 80)

paginas = []
for nombre_pdf in DOCUMENTOS_PDF:
    loader = PyPDFLoader(nombre_pdf)
    paginas_pdf = loader.load()
    texto_completo = "\n".join(p.page_content for p in paginas_pdf)
    paginas.append(Document(page_content=texto_completo, metadata={"source": nombre_pdf}))

splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=300)
fragmentos = splitter.split_documents(paginas)

print(f"\nTotal de fragmentos generados: {len(fragmentos)}\n")
for i, frag in enumerate(fragmentos):
    origen = frag.metadata.get("source", "?")
    print(f"--- Fragmento {i} | origen: {origen} | {len(frag.page_content)} caracteres ---")
    print(frag.page_content[:200].replace("\n", " "))
    print("...")
    print()

print("=" * 80)
print("PASO 2: Creando el índice de búsqueda (FAISS)...")
print("=" * 80)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
base_vectorial = FAISS.from_documents(fragmentos, embeddings)
retriever = base_vectorial.as_retriever(search_kwargs={"k": 8})
print("Índice listo.\n")

PREGUNTAS_A_PROBAR = [
    "¿Qué proyectos tengo que entregar este semestre, de qué materias y cuándo?",
    "¿Qué tipos de becas ofrece el instituto y cuáles son los requisitos de cada una?",
    "¿Quién imparte Química II y en qué horario da asesorías?",
]

for pregunta in PREGUNTAS_A_PROBAR:
    print("=" * 80)
    print("PASO 3: Probando la pregunta:", pregunta)
    print("=" * 80)

    docs_recuperados = retriever.invoke(pregunta)
    print(f"\nFragmentos recuperados para esta pregunta: {len(docs_recuperados)}")
    for d in docs_recuperados:
        origen = d.metadata.get("source", "?")
        print(f"  - {origen} ({len(d.page_content)} caracteres)")

print("\n\n")
print("=" * 80)
print("PASO 4: Probando la cadena completa (con el modelo de lenguaje)")
print("=" * 80)

llm = ChatCohere(model="command-a-03-2025", temperature=0)

plantilla_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Eres el agente académico del Instituto Educativo Horizonte. "
        "Responde la pregunta del estudiante usando SOLO la información del "
        "siguiente contexto. Si la respuesta involucra una lista o tabla "
        "(por ejemplo, materias, fechas, tipos de beca o periodos de examen), "
        "incluye TODOS los elementos relevantes que aparezcan en el contexto, "
        "no solo algunos. Antes de responder, revisa TODO el contexto en busca de "
        "cualquier sección relacionada con el tema, aunque no forme parte de la "
        "misma lista o tabla (por ejemplo, si preguntan sobre proyectos a entregar, "
        "considera tanto los proyectos por materia como el proyecto final "
        "institucional, si ambos aparecen). Si la información no está en el "
        "contexto, dilo claramente.\n\n"
        "Contexto:\n{context}\n\n"
        "Pregunta: {question}\n"
        "Respuesta completa:"
    ),
)

agente = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": plantilla_prompt},
)

for pregunta in PREGUNTAS_A_PROBAR:
    resultado = agente.invoke({"query": pregunta})
    print("\n❓ PREGUNTA:", pregunta)
    print("📄 Documentos usados:", [d.metadata.get("source", "?") for d in resultado["source_documents"]])
    print("💬 RESPUESTA:", resultado["result"])
    print("-" * 80)
