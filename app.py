import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_cohere import ChatCohere
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

load_dotenv()  # lee el archivo .env de esta carpeta y carga sus variables (por ejemplo COHERE_API_KEY)

st.set_page_config(page_title="Agente Académico - Instituto Horizonte", page_icon="📚")
st.title("📚 Agente Académico — Instituto Educativo Horizonte")

DOCUMENTOS_PDF = [
    "manual_academico_instituto_horizonte.pdf",
    "reglamento_del_estudiante.pdf",
    "politica_reembolso_matriculas.pdf",
    "faq_cursos_certificados.pdf",
    "guia_uso_plataforma.pdf",
    "programa_becas_afiliados.pdf",
]

MENSAJE_BIENVENIDA = (
    "¡Hola! 👋 Soy el agente académico del Instituto Educativo Horizonte. "
    "Puedo ayudarte a resolver dudas académicas sobre tus materias, el reglamento, "
    "reembolsos, certificados, la plataforma y becas. ¿Qué deseas consultar hoy?"
)

MENSAJE_NO_ACADEMICA = (
    "Esa no es una pregunta académica, así que no puedo responderla. 🙏 "
    "Solo puedo ayudarte con dudas del Instituto Horizonte: materias, calendario, "
    "profesores, proyectos, calificaciones, reglamento, reembolsos, certificados, "
    "la plataforma o becas."
)

# Etiqueta del botón -> pregunta real que se envía al agente
OPCIONES_RAPIDAS = {
    "📅 Fechas de inscripción / calendario": "¿Cuáles son las fechas importantes del calendario escolar de este semestre?",
    "📝 Dudas de calificaciones": "¿Dónde consulto mis calificaciones y qué hago si no estoy de acuerdo con una?",
    "🧪 Fechas de exámenes": "¿Cuándo son los periodos de exámenes parciales y finales?",
    "📂 Proyectos y entregas": "¿Qué proyectos tengo que entregar este semestre, de qué materias y cuándo?",
    "💰 Becas y afiliados": "¿Qué tipos de becas ofrece el instituto y cuáles son los requisitos de cada una?",
    "💳 Reembolso de matrícula": "¿Cómo solicito el reembolso de mi matrícula y qué porcentaje me corresponde?",
}

# Plantilla de instrucción personalizada: reemplaza el prompt genérico que usa
# RetrievalQA por defecto, exigiendo explícitamente respuestas completas
# (el modelo ya recibe casi todo el contenido, pero sin esta instrucción
# tendía a resumir u omitir elementos de listas y tablas).
PLANTILLA_PROMPT = PromptTemplate(
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


@st.cache_resource
def cargar_llm():
    return ChatCohere(model="command-a-03-2025", temperature=0)


@st.cache_resource
def cargar_agente():
    """Carga los 6 PDFs, los indexa con RAG (embeddings + FAISS) y arma la
    cadena de preguntas y respuestas.

    Nota de diseño: PyPDFLoader crea un fragmento POR PÁGINA del PDF antes de
    dividir, y el splitter nunca junta páginas distintas entre sí (solo divide
    una página si ella sola excede chunk_size). Por eso unimos primero todas
    las páginas de cada PDF en un solo texto, y ya sobre ese texto completo
    aplicamos el splitter — así chunk_size sí agrupa el documento completo
    (o lo divide en 2-3 partes si es muy largo), en vez de quedar partido
    por página sin importar qué tan corta sea cada una."""
    documentos_completos = []
    for nombre_pdf in DOCUMENTOS_PDF:
        loader = PyPDFLoader(nombre_pdf)
        paginas_pdf = loader.load()
        texto_completo = "\n".join(p.page_content for p in paginas_pdf)
        documentos_completos.append(
            Document(page_content=texto_completo, metadata={"source": nombre_pdf})
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=300)
    fragmentos = splitter.split_documents(documentos_completos)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    base_vectorial = FAISS.from_documents(fragmentos, embeddings)

    # k=8: con pocos fragmentos grandes en total, esto cubre prácticamente
    # todos los documentos relevantes para cualquier pregunta
    retriever = base_vectorial.as_retriever(search_kwargs={"k": 8})

    return RetrievalQA.from_chain_type(
        llm=cargar_llm(),
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PLANTILLA_PROMPT},
    )


def es_pregunta_academica(pregunta: str) -> bool:
    """Le pregunta al mismo modelo de lenguaje si la pregunta del usuario es
    académica/administrativa (relacionada con los documentos del instituto)
    o no, antes de gastar una búsqueda completa."""
    instruccion = (
        "Responde ÚNICAMENTE con la palabra SI o la palabra NO, sin explicaciones "
        "ni signos de puntuación adicionales.\n\n"
        "¿La siguiente pregunta de un estudiante trata sobre temas académicos o "
        "administrativos del instituto (materias, profesores, horarios de asesoría, "
        "calendario escolar, exámenes, proyectos, calificaciones, reglamento, "
        "reembolsos, certificados, uso de la plataforma o becas)?\n\n"
        f'Pregunta: "{pregunta}"'
    )
    respuesta = cargar_llm().invoke(instruccion)
    return respuesta.content.strip().upper().startswith("SI")


# Si falta la clave de API, avisamos con un mensaje claro en vez de que truene sin explicación
if not os.environ.get("COHERE_API_KEY"):
    st.error(
        "No encontré la variable de entorno COHERE_API_KEY. "
        "Crea un archivo `.env` en esta misma carpeta (usa `.env.example` como base) "
        "con la línea `COHERE_API_KEY=tu_clave`, guarda el archivo y vuelve a ejecutar "
        "`streamlit run app.py`."
    )
    st.stop()

archivos_faltantes = [f for f in DOCUMENTOS_PDF if not os.path.exists(f)]
if archivos_faltantes:
    st.error(
        "Faltan estos archivos en la carpeta (deben estar junto a app.py): "
        + ", ".join(archivos_faltantes)
    )
    st.stop()

with st.spinner("Preparando el agente (esto solo tarda la primera vez)..."):
    agente = cargar_agente()

# --- Historial de la conversación ---
# Empieza con el saludo de bienvenida del agente.
if "mensajes" not in st.session_state:
    st.session_state.mensajes = [{"role": "assistant", "content": MENSAJE_BIENVENIDA}]

# Muestra todo el historial acumulado hasta ahora
for m in st.session_state.mensajes:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m.get("fuentes"):
            with st.expander("Ver documentos usados para responder"):
                for doc in m["fuentes"]:
                    origen = doc.metadata.get("source", "?")
                    st.markdown(f"— *{origen}*")

# --- Opciones rápidas (siempre visibles, para usarlas en cualquier momento) ---
pregunta_boton = None
st.write("O elige una opción rápida:")
columnas = st.columns(2)
for i, (etiqueta, pregunta_real) in enumerate(OPCIONES_RAPIDAS.items()):
    if columnas[i % 2].button(etiqueta, use_container_width=True, key=f"opcion_{i}"):
        pregunta_boton = pregunta_real

# --- Caja de texto libre ---
pregunta_usuario = st.chat_input("Escribe tu pregunta...")

pregunta_final = pregunta_boton or pregunta_usuario

if pregunta_final:
    st.session_state.mensajes.append({"role": "user", "content": pregunta_final})

    # Los botones de opciones rápidas ya son preguntas académicas por diseño,
    # así que solo clasificamos lo que el usuario escribió libremente.
    if pregunta_boton or es_pregunta_academica(pregunta_final):
        with st.spinner("Buscando en los documentos del instituto..."):
            resultado = agente.invoke({"query": pregunta_final})
        st.session_state.mensajes.append({
            "role": "assistant",
            "content": resultado["result"],
            "fuentes": resultado["source_documents"],
        })
    else:
        st.session_state.mensajes.append({"role": "assistant", "content": MENSAJE_NO_ACADEMICA})

    st.rerun()
