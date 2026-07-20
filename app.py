import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_cohere import ChatCohere
from langchain_classic.chains import RetrievalQA

load_dotenv()  # lee el archivo .env de esta carpeta y carga sus variables (por ejemplo COHERE_API_KEY)

st.set_page_config(page_title="Agente Académico - Instituto Horizonte", page_icon="📚")
st.title("📚 Agente Académico — Instituto Educativo Horizonte")

NOMBRE_PDF = "manual_academico_instituto_horizonte.pdf"

MENSAJE_BIENVENIDA = (
    "¡Hola! 👋 Soy el agente académico del Instituto Educativo Horizonte. "
    "Puedo ayudarte a resolver dudas académicas sobre tus materias, con base "
    "en el manual del semestre. ¿Qué deseas consultar hoy?"
)

MENSAJE_NO_ACADEMICA = (
    "Esa no es una pregunta académica, así que no puedo responderla. 🙏 "
    "Solo puedo ayudarte con dudas académicas del Instituto Horizonte: calendario, "
    "profesores, horarios de asesoría, proyectos, exámenes o calificaciones."
)

# Etiqueta del botón -> pregunta real que se envía al agente
OPCIONES_RAPIDAS = {
    "📅 Fechas de inscripción / calendario": "¿Cuáles son las fechas importantes del calendario escolar de este semestre?",
    "📝 Dudas de calificaciones": "¿Dónde consulto mis calificaciones y qué hago si no estoy de acuerdo con una?",
    "🧪 Fechas de exámenes": "¿Cuándo son los periodos de exámenes parciales y finales?",
    "📂 Proyectos y entregas": "¿Qué proyectos tengo que entregar este semestre, de qué materias y cuándo?",
}


@st.cache_resource
def cargar_llm():
    return ChatCohere(model="command-a-03-2025", temperature=0)


@st.cache_resource
def cargar_agente():
    """Carga el PDF, lo indexa y arma la cadena de preguntas y respuestas.
    Se ejecuta una sola vez gracias a @st.cache_resource, aunque varias
    personas usen la app al mismo tiempo."""
    loader = PyPDFLoader(NOMBRE_PDF)
    paginas = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    fragmentos = splitter.split_documents(paginas)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    base_vectorial = FAISS.from_documents(fragmentos, embeddings)

    retriever = base_vectorial.as_retriever(search_kwargs={"k": 4})

    return RetrievalQA.from_chain_type(
        llm=cargar_llm(), retriever=retriever, return_source_documents=True
    )


def es_pregunta_academica(pregunta: str) -> bool:
    """Le pregunta al mismo modelo de lenguaje si la pregunta del usuario es
    académica (relacionada con el manual del instituto) o no, antes de
    gastar una búsqueda completa en el documento."""
    instruccion = (
        "Responde ÚNICAMENTE con la palabra SI o la palabra NO, sin explicaciones "
        "ni signos de puntuación adicionales.\n\n"
        "¿La siguiente pregunta de un estudiante trata sobre temas académicos "
        "del semestre (materias, profesores, horarios de asesoría, calendario "
        "escolar, exámenes, proyectos, calificaciones o trámites escolares)?\n\n"
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

if not os.path.exists(NOMBRE_PDF):
    st.error(
        f"No encontré el archivo '{NOMBRE_PDF}' en esta carpeta. "
        "Colócalo junto a app.py antes de iniciar la aplicación."
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
            with st.expander("Ver fragmentos del manual usados para responder"):
                for doc in m["fuentes"]:
                    st.markdown(f"— *Página {doc.metadata.get('page', '?')}*")
                    st.caption(doc.page_content[:300] + "...")

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
        with st.spinner("Buscando en el manual académico..."):
            resultado = agente.invoke({"query": pregunta_final})
        st.session_state.mensajes.append({
            "role": "assistant",
            "content": resultado["result"],
            "fuentes": resultado["source_documents"],
        })
    else:
        st.session_state.mensajes.append({"role": "assistant", "content": MENSAJE_NO_ACADEMICA})

    st.rerun()
