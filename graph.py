import base64
import io
import json
import os
import time
from typing import Annotated, ClassVar, Literal, Optional, Sequence, TypedDict
import urllib3
from functools import partial

import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image
from pymongo import MongoClient
from pydantic import BaseModel, Field

from langchain.agents import AgentType, initialize_agent
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.store.mongodb import MongoDBStore

import configuration

load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3
)


# MongoDB connection
mongo_username = os.getenv('MONGO_USERNAME')
mongo_password = os.getenv('MONGO_PASSWORD')
MONGO_URL = f"mongodb+srv://{mongo_username}:{mongo_password}@cluster0.vzjzms8.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URL)
db = client["agent_rag_db"]
collection = db["user_memory"]
across_thread_memory = MongoDBStore(collection=collection)

# Short-term memory for within-thread
within_thread_memory = MemorySaver()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    requires_agent: bool 
    requires_add_data: bool 
    input_img : str 
    output_img : str 
    requires_output_img : bool 
    rag_context: str
    research_answer: str 
    final_answer: str
    planned : list[str] 
    query: str


 ### Image Loader


import os
import requests
from io import BytesIO
import PIL 

def load_image(input_data):

    try:

        if isinstance(input_data, PIL.Image.Image):
            return input_data

        if isinstance(input_data, str) and input_data.startswith(("http://", "https://")):
            response = requests.get(input_data, timeout=10)
            response.raise_for_status()
            return PIL.Image.open(BytesIO(response.content)).convert("RGB")

        if isinstance(input_data, str) and os.path.exists(input_data):
            # Image.open yerine tam yolunu belirtiyoruz:
            return PIL.Image.open(input_data).convert("RGB")

        return None

    except Exception as e:
        print(f"Hata: {e}")
        return None


 ### Rag system


from rag_system import RagSystem
rag_system = RagSystem()


embeddings_model = rag_system.embeddings_model
vectorstore = rag_system.vectorstore
blip_processor = rag_system.blip_processor
blip_model = rag_system.blip_model

def generate_image_description(image: Image.Image):
    """BLIP ile görsel açıklaması üret"""
    try:
        inputs = blip_processor(image, return_tensors="pt")
        out = blip_model.generate(**inputs, max_length=100, num_beams=5)
        description = blip_processor.decode(out[0], skip_special_tokens=True)
        return description
    except Exception as e:
        print(f"❌ Görsel açıklama hatası: {e}")
        return "Görsel açıklama üretilemedi"

def vector_search_text_img(query_text, image, k=3):
    """Text + Görsel açıklaması → Ortak embedding ile arama"""
    # Text embedding
    text_emb = embeddings_model.embed_query(query_text)

    # Görsel açıklaması + embedding
    image_desc = generate_image_description(image)
    img_emb = embeddings_model.embed_query(image_desc)

    # Ortalama al (basit fusion)
    combined_emb = (np.array(text_emb) + np.array(img_emb)) / 2

    # Vector arama
    return vectorstore.similarity_search_by_vector(combined_emb.tolist(), k=k)

def vector_search_img(image, k=3):
    """Sadece görsel açıklaması ile arama"""
    # Görsel açıklaması üret
    image_desc = generate_image_description(image)
    
    # Embedding al
    img_emb = embeddings_model.embed_query(image_desc)

    # Vector arama
    return vectorstore.similarity_search_by_vector(img_emb, k=k)



 #### Decision agent


class DecisionOutput(BaseModel):
    requires_agent: bool = Field(description="Determine whether the query can be answered directly or requires action (research, need external tool, etc).")
    answer: Optional[str] = Field(default=None, description="The response should be None if the user query requires agent; otherwise, provide a direct answer.")


decision_making_prompt = """
Based on the user's query, decide whether you need to use an agent (external tool, research, etc.) 
or if you can answer the question directly.

**Memory(it may be empty):** 
{memory} 


- Answer the question directly only for simple conversational queries, such as "How are you?" or "What's your name?". Also if you can answer the question directly.

- If you will answer then use **Memory** to personalize your response based on memory. Use personalization especially in:
    * Greetings and transitions
    * Help or guidance tailored to tools and frameworks the user uses
    * Follow-up messages that continue from past context
        
- For all other queries, ignore direct answering and decide if an agent is needed.

"""

def decision_node(state: AgentState, config: RunnableConfig, store: MongoDBStore):
    
    state["output_img"] = ""  
    state["requires_output_img"] = False
    state["rag_context"] = ""
    state["research_answer"] = ""
    state["planned"] = []
    state["requires_add_data"] = False


    # --- 1. Kullanıcı memory'sini MongoDB'den al ---
    configurable = configuration.Configuration.from_runnable_config(config)

    # Get the user ID from the config
    user_id = configurable.user_id
    namespace = ("memory", user_id)
    key = "user_memory"
    existing_memory = store.get(namespace, key)
    existing_memory_content = existing_memory.value.get('memory') if existing_memory else "No existing memory found."

    # --- 2. LLM payload'u oluştur ---
    current_messages = state["messages"]   # liste olarak al
    last_user_message = current_messages[-1]   # son mesaj
    input_img = load_image(state.get("input_img", None))

    # Memory'yi prompt içine ekle
    prompt_with_memory = decision_making_prompt.format(memory=existing_memory_content)
    llm_payload = [SystemMessage(content=prompt_with_memory)]

    # --- 3. Mesaj geçmişini ve varsa resmi ekle ---
    if input_img is None:
        llm_payload.append(last_user_message)   # sadece son mesaj
    else:
        # Önceki mesajlar (sonuncusu hariç)
        history = current_messages[:-1]
        llm_payload.extend(history)

        # Son kullanıcı mesajı
        last_user_msg_text = ""
        if isinstance(last_user_message.content, str):
            last_user_msg_text = last_user_message.content
        elif isinstance(last_user_message.content, list):
            for part in last_user_message.content:
                if part.get("type") == "text":
                    last_user_msg_text = part.get("text", "")
                    break

        # Resmi base64'e çevir
        buffered = BytesIO()
        input_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        multimodal_message = HumanMessage(content=[
            {"type": "text", "text": last_user_msg_text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
        ])
        llm_payload.append(multimodal_message)

    # --- 4. LLM'i çağır ---
    decision_llm = llm.with_structured_output(DecisionOutput)
    response: DecisionOutput = decision_llm.invoke(llm_payload)

    if response.answer:
        state["final_answer"] = response.answer
    state["requires_agent"] = response.requires_agent
    state["input_img"] = input_img
    print("######################")
    print(state["input_img"])
    print("######################")
    return state
########################################################################


 #### Decision Router


def decision_router(state: AgentState):
    if state["requires_agent"]:
        return "planning"
    else:
        return "end"


 ### Planner agent



class PlannerState(BaseModel):
    planned: list[Literal["Researcher Agent", "RAG Query Agent"]] = Field(description="Plan based on the user's query. List each agent as a separate item.")
    requires_add_data: bool = Field(description="Does the user want to add data to the RAG system? (True if yes, False if no)")



class AddingData(BaseModel):
    img_is_available: bool = Field(description="Indicates whether the user query specifies that an image will be included in the data to be added (True if yes, False if no).")
    data_to_add: str = Field(description="The corrected data that will be added to the RAG system.")




planner_prompt = """
You are an agency planner. Your job is to decide which agents are needed based on the user's query.

Output Format (MUST follow exactly):
{
  "planned": ["Researcher Agent" or "RAG Query Agent", ...],
  "requires_add_data": true or false
}

Available Agents:
* Researcher Agent: Gathers and synthesizes information from various sources (e.g., the web).
* RAG Query Agent: Retrieves relevant, specific information for clothing-related data (Except adding data).

EXAMPLES:
1. Query: Find data on blue jeans and research what blue pants are made of. 
   -> {"planned": ["Researcher Agent", "RAG Query Agent"], "requires_add_data": false}

2. Query: **IMAGE** What season is this jacket from? 
   -> {"planned": ["RAG Query Agent"], "requires_add_data": false}

3. Query: Bring me image of blue jean? 
   -> {"planned": ["RAG Query Agent"], "requires_add_data": false}

4. Query: When was Google founded? 
   -> {"planned": ["Researcher Agent"], "requires_add_data": false}

5. Query: Find me black Puma shoes. 
   -> {"planned": ["RAG Query Agent"], "requires_add_data": false}

6. Query: Add to RAG system that white watches. 
   -> {"planned": [], "requires_add_data": true}
"""



adding_prompt=""""
Your task is to determine what corrected data should be added to the RAG system and whether the user specified that an image will be included.

- For `img_is_available`: Set it to True only if the user explicitly mentions that an image will be added; otherwise, set it to False.
- For `data_to_add`: Extract and provide only the corrected information that needs to be stored in the RAG system.
- Base your answer only on the parts of the input that are relevant to adding data into the RAG system, and ignore any unrelated or irrelevant text.

Data must english.
"""



def planner_node(state: AgentState):
    planner_llm = llm.with_structured_output(PlannerState)

    system_prompt = SystemMessage(content=planner_prompt)

    response : PlannerState = planner_llm.invoke([system_prompt, state["messages"][-1].content])

    if response.requires_add_data:

        adding_llm = llm.with_structured_output(AddingData)

        system_prompt = SystemMessage(content=adding_prompt)

        response_add : AddingData = adding_llm.invoke([system_prompt, state["messages"][-1].content])
        print(f"New Data:{response_add.data_to_add}")
        
        if response.img_is_available:
            rag_system.add_data(response_add.data_to_add, load_image(state.get("input_img")))
        else:
            rag_system.add_data(response_add.data_to_add)

    return {"planned": response.planned, "requires_add_data": response.requires_add_data}


class ReWritingState(BaseModel):
    # Kısa, temizlenmiş ve retrieval-odaklı sorgu (1-2 cümle)
    query: str = Field(description="Concise, cleaned query for retrieval (1-2 sentences).")
    # HyDE: retrieval için oluşturulmuş ayrıntılı hipotez-belgesi (zorunlu, en az ~150-250 kelime önerilir)
    hyde_document: str = Field(description="Detailed hypothetical document (HyDE) that directly answers the query and provides retrieval-friendly keywords, suggested search queries, entity/date hints, and a short summary.")
    # Kullanıcı açıkça görüntü istemiş mi?
    requires_output_img: bool = Field(description="True if user explicitly requests an image/infographic/etc., otherwise False.")



query_rewrite_prompt = PromptTemplate(
    input_variables=["original_query"],
    template="""
You are a rewriting agent for a RAG system specialized in clothing and fashion-related data. 
Your task is to extract **only the part of the user's input that is relevant for this RAG system**, 
ignoring any other unrelated requests (e.g., web search, general knowledge or non-fashion topics).

Produce three structured items:
1) query: a concise, search-friendly version of the user's RAG-relevant request (1-2 sentences), focused only on clothing/fashion. 
2) requires_output_img: boolean — True only if the user explicitly asks for an image, diagram, or visual of clothing/fashion items.
3) hyde_document: a standalone, retrieval-friendly hypothetical document (HyDE) fully answering the extracted query, containing:
   - SUMMARY: 1-2 sentence concise answer
   - KEY FACTS: 3-6 facts or claims, mark uncertain items with [VERIFY]
   - SUGGESTED RETRIEVAL QUERIES: 5-8 variant search queries or keywords
   - KEYWORDS / SYNONYMS / ALIASES: compact list
   - DOCUMENT TYPES: e.g., "catalog", "product description", "fashion article"

***** IMPORTANT RULES *****
- Always produce a HyDE for the extracted RAG query.
- If the user mentions multiple topics, **ignore unrelated topics**. Only include fashion/clothing parts.
- requires_output_img = True only if user explicitly requests a visual for the clothing/fashion query.
- Output only JSON with exactly these keys: query, requires_output_img, hyde_document.

User original input (raw): {original_query}
""")



def rewriting_node(state: AgentState):
    # Kullanıcının son mesajını al
    original_query = state["messages"][-1].content
    
    # Prompt'u doldur
    prompt = query_rewrite_prompt.format(original_query=original_query)
    
    rewriting_llm: ReWritingState = llm.with_structured_output(ReWritingState)

    # LLM'e gönder
    response = rewriting_llm.invoke(prompt)
    
    if response and response.hyde_document:
        result = response.hyde_document
    else:
        result= response.query

    return {"query": result, "requires_output_img" : response.requires_output_img}




def rag_query(state: AgentState):
    # Kullanıcının son mesajını al
    original_query = state["query"]
    # Görsel açıklaması varsa, arama yap

        
    if state["requires_output_img"]:
        if state.get("input_img", ""):
            results = vector_search_text_img(original_query, state["input_img"], k=1)
            print("Text and image search with output image...")
            return {"rag_context": results[0].page_content, "output_img": results[0].metadata["link"]}
        else:
            results = rag_system.similarity_search_with_score(original_query, k=1)
            print("Text search with output image...")
            return {"rag_context": results[0][0].page_content, "output_img": results[0][0].metadata["link"]}

    else:
        if state.get("input_img", ""):
            results = vector_search_text_img(original_query, state["input_img"], k=1)
            print("Just text and image search ...")
            return {"rag_context": results[0].page_content}
        else:
            results = rag_system.similarity_search_with_score(original_query, k=1)
            print("Just text search ...")
            return {"rag_context": results[0][0].page_content}        


 ### Researcher



def researcher_node(state: AgentState):
    original_query = state["messages"][-1].content
    
    tools = [DuckDuckGoSearchRun()]

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        agent_kwargs={
            "prefix": (
                "You are a research assistant. "
                "Ignore irrelevant or non-researchable parts of the query. "
                "Provide a clear and concise answer only for the researched part."
                "If need tools you can use duckduckgo for web search."
            )
        }
    )

    response = agent.invoke({"input": original_query})
    return {"research_answer": response["output"]}


 ### Router 


def router(state: AgentState):
    planned = state.get("planned")

    if not planned:
        return "generate_answer"  
    elif "Researcher Agent" in planned[0]:
        planned.pop(0)
        return "researcher"
    elif "RAG Query Agent" in planned[0]:
        planned.pop(0)
        return "rewriting"
    else:
        return "generate_answer"


 ### Generate Answer


generate_prompt = """
You are an AI assistant tasked with generating a clear, accurate, and helpful response for the user. 
Using the user's input, formulate your response based on the provided 'context' and 'another answer'.

**Output Image is exist(It may be empty)(No need to display image):** {requires_output_img}

**User Input:**
{user_input}

**Context(It may be empty)(It may related to user input so use this context to answer user input.):**
{context}

**Researcher Answer(It may be empty):**
{researcher_answer}

**Memory(it may be empty):**
{memory}

If you have memory for this user, use it to personalize your responses. Use personalization especially in:
    * Greetings and transitions
    * Help or guidance tailored to tools and frameworks the user uses
    * Follow-up messages that continue from past context
    
The answer should be in the same language as the user input.
Generate the best possible response.
"""


def generate_answer(state: AgentState, config: RunnableConfig, store: MongoDBStore):
    # --- 1. Retrieve user memory from MongoDB ---
    user_id = config["configurable"]["user_id"]
    namespace = ("memory", user_id)
    key = "user_memory"
    existing_memory = store.get(namespace, key)
    existing_memory_content = existing_memory.value.get('memory') if existing_memory else "No existing memory found."

    # --- 2. Create prompt template  ---
    prompt_template = ChatPromptTemplate.from_template(generate_prompt)

    requires_output_img = state.get("requires_output_img", "")

    # --- 3. Format ---
    prompt = prompt_template.format_messages(
        user_input=state["messages"][-1].content,
        context=state.get("rag_context", ""),
        researcher_answer=state.get("research_answer", ""),
        requires_output_img=requires_output_img,
        memory=existing_memory_content  
    )

    # --- 4. Invoke LLM ---
    response = llm.invoke(prompt)

    return {"final_answer": response.content, "output_img": state.get("output_img", "")}


class MemoryState(BaseModel):
    memory: str = Field(description="User profile information collected from chat history")


create_memory_prompt = """"You are collecting information about the user to personalize your responses.

CURRENT USER INFORMATION:
{memory}

INSTRUCTIONS:
1. Review the chat history below carefully
2. Identify new information about the user, such as:
   - Personal details (name, location, e.g.)
   - Preferences (likes, dislikes)
   - Interests and hobbies
   - Past experiences
   - Goals or future plans
3. Merge any new information with existing memory
4. Format the memory as a clear, bulleted list
5. If new information conflicts with existing memory, keep the most recent version

Remember: Only include factual information directly stated by the user. Do not make assumptions or inferences.
Important: Do NOT include summaries like "no update needed". Dont add your command.
If no new data is available in chat history below, return the user information unmodified. Like this:
{memory}

Update the user information based on the chat history below, maintaining the original structure:
"""

def write_memory(state: AgentState, config: RunnableConfig, store: MongoDBStore):

    configurable = configuration.Configuration.from_runnable_config(config)

    # Get the user ID from the config
    user_id = configurable.user_id
    namespace = ("memory", user_id)
    key = "user_memory"
    existing_memory = store.get(namespace, key)
    existing_memory_content = existing_memory.value.get('memory') if existing_memory else "No existing memory found."
    system_msg = create_memory_prompt.format(memory=existing_memory_content)

    memory_llm: MemoryState = llm.with_structured_output(MemoryState)

    new_memory = memory_llm.invoke([SystemMessage(content=system_msg)] + state["messages"])
    store.put(namespace, key, {"memory": new_memory.memory})



### Graph



workflow = StateGraph(AgentState, config_schema=configuration.Configuration)

workflow.add_node("decision_node", partial(decision_node, store=across_thread_memory))
workflow.add_node("planner_node", planner_node)
workflow.add_node("rewriting_node", rewriting_node)
workflow.add_node("rag_query", rag_query)
workflow.add_node("researcher_node", researcher_node)
workflow.add_node("generate_answer", partial(generate_answer, store=across_thread_memory))
workflow.add_node("write_memory", partial(write_memory, store=across_thread_memory))

# Set the entry point of the graph
workflow.add_edge(START, "decision_node")

workflow.add_conditional_edges(
    "decision_node",
    decision_router,
    {"end": "write_memory", "planning": "planner_node"}
)

workflow.add_conditional_edges(
    "planner_node",
    router,
    {"researcher": "researcher_node", "rewriting": "rewriting_node", "generate_answer": "generate_answer"}
)

workflow.add_conditional_edges(
    "researcher_node",
    router,
    {"rewriting": "rewriting_node", "generate_answer": "generate_answer"}
)

workflow.add_conditional_edges(
    "rag_query",
    router,
    {"researcher": "researcher_node", "generate_answer": "generate_answer"}
)

workflow.add_edge("rewriting_node", "rag_query")
workflow.add_edge("generate_answer", "write_memory")
workflow.add_edge("write_memory", END)


graph = workflow.compile()


