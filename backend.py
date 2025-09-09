"""
Flask Backend for ZeeVice AI Frontend Integration
Integrated with hybrid search and GraphRAG functionality
"""

from flask import Flask, request, jsonify, session
from collections import defaultdict
from neo4j import GraphDatabase
from dotenv import dotenv_values
from openai import OpenAI
import json
import chromadb
from chromadb.utils import embedding_functions
import time
from datetime import datetime, timedelta
from threading import Event
import os
from flask_cors import CORS

app = Flask(__name__)
app.secret_key = os.urandom(24) # Needed for Flask sessions

# Enable CORS for the React frontend
CORS(app)


# Load as a dictionary
# Ensure file.env is in the same directory as app.py or provide full path
config = dotenv_values("file.env")

# --- Global Initialization (outside of request context) ---
# CHROMA DB SETUP
PATH_TO_CHROMA = config.get('PATH_TO_CHROMA')
COLLECTION_NAME = config.get('COLLECTION_NAME')
EMBEDDING_MODEL = config.get('EMBEDDING_MODEL_NAME')

try:
    chroma_client = chromadb.PersistentClient(path=PATH_TO_CHROMA)
    chroma_collection = chroma_client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
    )
    # Initialize a separate client/collection for the product cache
    chroma_cache_client = chromadb.PersistentClient(path=os.path.join(PATH_TO_CHROMA, "product_cache_db"))
    chroma_cache_collection = chroma_cache_client.get_or_create_collection(
        name="product_cache",
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    )
except Exception as e:
    print(f"Error initializing ChromaDB: {e}")
    chroma_collection = None
    chroma_cache_collection = None

# NEO4J SETUP
URI = config.get('URI')
USERNAME = config.get('USERNAME')
PASSWORD = config.get('PASSWORD')

try:
    neo4j_driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    neo4j_driver.verify_connectivity() # Test connection
except Exception as e:
    print(f"Error connecting to Neo4j: {e}")
    neo4j_driver = None

# LLAMA-4 SETUP
API_KEY = config.get('API_KEY')
BASE_URL = config.get('BASE_URL')
LLM = config.get('LLM')

try:
    openai_agent = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    openai_agent = None


# Token Tracker (instance per application, not per request)
class RateLimiter:
    def __init__(self, tpm_limit=25000):
        self.tpm_limit = tpm_limit
        self.tokens_used = 0
        self.reset_time = datetime.now() + timedelta(minutes=1)
        self.countdown_event = Event()
        self.countdown_thread = None

    def _start_countdown(self, wait_seconds: int):
        # For a web app, printing to console isn't ideal for user feedback.
        # This would typically be handled by frontend polling or websockets.
        # Keeping it for now for internal logging/debugging.
        # print(f"Rate limit cooldown: {wait_seconds} seconds")
        # The original countdown logic is blocking, which is bad for a web server.
        # In a real-world scenario, you'd use a non-blocking approach (e.g., async tasks,
        # or return a 429 Too Many Requests and let the client retry).
        # For this example, we'll keep the blocking sleep for simplicity, but be aware.
        time.sleep(wait_seconds)


    def wait_if_needed(self, response):
        new_tokens = response.usage.total_tokens
        now = datetime.now()

        if now > self.reset_time:
            self.tokens_used = 0
            self.reset_time = now + timedelta(minutes=1)

        estimated_tpm = self.tokens_used + new_tokens
        if estimated_tpm > self.tpm_limit * 0.8:
            wait_seconds = max(0, (self.reset_time - now).total_seconds())

            if wait_seconds > 0:
                self._start_countdown(int(wait_seconds))
                self.countdown_event.set()

                self.tokens_used = 0
                self.reset_time = datetime.now() + timedelta(minutes=1)

        self.tokens_used += new_tokens
        return self.tokens_used

app_rate_limiter = RateLimiter()


# --- Helper Functions (adapted for Flask context) ---

def extract_structured_feats(user_query: str):
    messages = [{
        "role": "system",
        "content": (
            """From the given query, extract 5 features:
        1- category: smartphones or laptops, in this exact format
        2- brand: brand of the mentioned device in lowercase format
        3- store: [B.Tech, 2B Egypt, Noon, Raya] dont extract any other store
        4- price_min: minimum price in EGP
        5- price_max: maximum price in EGP

        If anything of these specs are not explicitly mentioned, deposition it as null
        Both "brand" and "store" should be in lists

        Return the specs in a json format
        Don't add any extra chit-chat.
        """
        )
    }, {"role": "user", "content": user_query}]

    resp = openai_agent.chat.completions.create(model=LLM, messages=messages, temperature=0.1)
    app_rate_limiter.wait_if_needed(resp)
    text = resp.choices[0].message.content.strip()
    json_str = text[text.find('{'): text.find('}') + 1]
    structured_feats = json.loads(json_str)

    structured_feats["brand"] = structured_feats.get("brand", [])
    structured_feats["store"] = structured_feats.get("store", [])
    return structured_feats


def identify_similarity_feats(user_query: str):
    messages = [{
        "role": "system",
        "content": (
            """From the given query, I need you to identify which of the following specs will be relevant to find
        similar smartphones:
        [display_size, display_type, ram, storage, main_camera, front_camera, battery, cores, network, os, sim]
        Return a json object that has each of the above specs with true if you believe it will be VITAL to find similar
        smartphones, else false.
        Don't add any extra chit-chat.
        """
        )
    }, {"role": "user", "content": user_query}]

    resp = openai_agent.chat.completions.create(model=LLM, messages=messages, temperature=0.1)
    app_rate_limiter.wait_if_needed(resp)
    text = resp.choices[0].message.content.strip()
    json_str = text[text.find('{'): text.rfind('}') + 1]
    similarity_feats = json.loads(json_str)
    return similarity_feats


def extract_unstructured_feats(user_query: str, only_keys=None):
    keys_str = ", ".join(only_keys) if only_keys else "display_size, display_type, ram, storage, main_camera, front_camera, battery, cores, network, os, sim"
    messages = [{
        "role": "system",
        "content": (
            f"From the query, extract concrete values for these specs if present: [{keys_str}]. "
            "Return JSON object with any present keys and short normalized values; omit keys that are absent. "
            "Examples: {'ram':'8 GB','display_type':'AMOLED','battery':'5000 mAh','cores':'octa core'}"
            "Make sure to wrap each spec name in double quotes."
            "Don't add any extra chit-chat."
        )
    }, {"role": "user", "content": user_query}]

    resp = openai_agent.chat.completions.create(model=LLM, messages=messages, temperature=0.1)
    app_rate_limiter.wait_if_needed(resp)
    text = resp.choices[0].message.content.strip()
    start, end = text.find('{'), text.find('}') + 1
    if start == -1 or end == 0:
        return {}
    unstructured_feats = json.loads(text[start:end])
    return unstructured_feats


def return_exact_ids(filterable_feats: dict, limit: int = 5000):
    if not neo4j_driver: return []
    cypher = ["MATCH (p:PRODUCT)"]
    where = []
    params = {}

    if filterable_feats.get('category'):
        cypher.append("-[:BELONG_TO]->(c:CATEGORY)")
        where.append("toLower(c.name) = $category")
        params["category"] = filterable_feats['category'].strip().lower()

    store = filterable_feats.get('store') or []
    if store:
        cypher.append(", (p)-[:SOLD_AT]->(s:STORE)")
        where.append("toLower(s.name) IN $stores")
        params["stores"] = [s.strip().lower() for s in store if s]

    brand = filterable_feats.get('brand') or []
    if brand:
        where.append("toLower(p.brand) IN $brands")
        params["brands"] = [b.strip().lower() for b in brand if b]

    price_min = float(filterable_feats.get('price_min') or 0)
    price_max = float(filterable_feats.get('price_max') or 9_999_999)
    where.append("p.price >= $pmin AND p.price <= $pmax")
    params["pmin"], params["pmax"] = price_min, price_max

    cypher_line = "".join(cypher)
    if where:
        cypher_line += " WHERE " + " AND ".join(where)
    cypher_line += " RETURN p.ID AS id LIMIT $limit"
    params["limit"] = limit

    with neo4j_driver.session() as session:
        result = session.run(cypher_line, **params)
        return [str(r["id"]) for r in result]


def return_similar_graph_ids(exact_ids: list, similarity_flags: dict, limit: int = 400, batch_size: int = 20):
    if not exact_ids or not neo4j_driver or not chroma_collection:
        return []

    cypher = "MATCH (p:PRODUCT) WHERE p.ID IN $ids RETURN properties(p) AS props"
    props_list = []
    with neo4j_driver.session() as session:
        res = session.run(cypher, ids=[int(pid) if str(pid).isdigit() else pid for pid in exact_ids])
        for r in res:
            props_list.append(r["props"])

    spec_texts = []
    for pr in props_list:
        parts = []
        for k, is_on in (similarity_flags or {}).items():
            if is_on and k in pr and pr[k]:
                parts.append(f"{k}: {pr[k]}")
        if parts:
            spec_texts.append(", ".join(parts))

    similar_ids = []
    remaining = min(limit, 400)
    for i in range(0, len(spec_texts), batch_size):
        if remaining <= 0:
            break
        batch = spec_texts[i:i + batch_size]
        n = min(batch_size, remaining)
        ch = chroma_collection.query(query_texts=batch, n_results=n)
        for hitlist in ch["ids"]:
            similar_ids.extend(hitlist)
        remaining -= sum(len(hitlist) for hitlist in ch["ids"])

    original_set = set(exact_ids)
    similar_ids = [str(x) for x in similar_ids if str(x) not in original_set]
    return list(dict.fromkeys(similar_ids))


def chroma_semantic(user_query: str, similarity_flags: dict, limit: int = 200):
    if not chroma_collection: return {}
    keys = [k for k, v in (similarity_flags or {}).items() if v]
    kv = extract_unstructured_feats(user_query, only_keys=keys)

    texts = [user_query]
    if kv:
        kv_str = ", ".join([f"{k}: {v}" for k, v in kv.items()])
        texts.append(kv_str)

    result = chroma_collection.query(query_texts=texts, n_results=min(limit, 200), include=["distances"])
    best = {}
    for q_i, hit_ids in enumerate(result["ids"]):
        dists = result["distances"][q_i]
        for j, pid in enumerate(hit_ids):
            sim = 1 - dists[j]
            pid = str(pid)
            if pid not in best or sim > best[pid]:
                best[pid] = sim

    return best


def normalize_semantic(scores: dict):
    if not scores:
        return {}
    mx = max(scores.values())
    if mx <= 0:
        return {k: 0.0 for k in scores}
    return {k: v / mx for k, v in scores.items()}


def score_and_rank(exact_graph_ids: list, similar_graph_ids: list, chroma_scores_raw: dict, top_k: int = 10):
    graph_scores = {}
    for pid in exact_graph_ids:
        graph_scores[str(pid)] = 1.0
    for pid in similar_graph_ids:
        pid = str(pid)
        graph_scores[pid] = max(graph_scores.get(pid, 0.0), 0.6)

    sem_norm = normalize_semantic({str(k): float(v) for k, v in (chroma_scores_raw or {}).items()})

    overlap = set(graph_scores) & set(sem_norm)
    if overlap:
        w_s, w_u = 0.6, 0.4
    else:
        w_s, w_u = 0.5, 0.5

    final_scores = defaultdict(float)
    for pid, s in graph_scores.items():
        final_scores[pid] += w_s * s
    for pid, u in sem_norm.items():
        final_scores[pid] += w_u * u

    for pid in overlap:
        final_scores[pid] += 0.08

    ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def hybrid_search(user_query: str, top_k: int = 10, graph_limit_exact=1500, graph_limit_sim=400, chroma_limit=200):
    if not neo4j_driver or not chroma_collection or not openai_agent:
        return {"error": "Backend services not initialized. Check server logs."}

    filt = extract_structured_feats(user_query)
    exact_ids = return_exact_ids(filt, limit=graph_limit_exact)

    sim_flags = identify_similarity_feats(user_query)

    similar_ids = return_similar_graph_ids(exact_ids, sim_flags, limit=graph_limit_sim)

    chroma_best = chroma_semantic(user_query, sim_flags, limit=chroma_limit)

    ranked = score_and_rank(exact_ids, similar_ids, chroma_best, top_k=top_k)

    top_ids = [int(pid) if str(pid).isdigit() else pid for pid, _ in ranked]
    cards = []
    if top_ids:
        cypher = "MATCH (p:PRODUCT) WHERE p.ID IN $ids RETURN properties(p) AS props"
        with neo4j_driver.session() as session:
            res = session.run(cypher, ids=top_ids)
            for r in res:
                props = r["props"]
                cards.append({
                    "id": str(props.get("ID")),
                    "title": props.get("title"),
                    "brand": props.get("brand"),
                    "price": props.get("price"),
                    "store": props.get("store"),
                    "url": props.get("product_url"),
                    "display_size": props.get("display_size"),
                    "display_type": props.get("display_type"),
                    "ram": props.get("ram"),
                    "storage": props.get("storage"),
                    "battery": props.get("battery"),
                    "cores": props.get("cores"),
                    "os": props.get("os"),
                    "sim": props.get("sim"),
                    "network": props.get("network"),
                    "main_camera": props.get("main_camera"),
                    "front_camera": props.get("front_camera"),
                })
    products = {
        "query": user_query,
        "filters": filt,
        "sim_flags": sim_flags,
        "ranked": ranked,
        "cards": cards
    }

    return products


def cache_builder(product_cache: list, user_query: str):
    if not chroma_cache_collection: return []
    ids = [str(hash(p)) for p in product_cache]

    existing_ids = set()
    # Check if collection has any documents before trying to get by IDs
    if chroma_cache_collection.count() > 0:
        try:
            existing_ids = set(chroma_cache_collection.get(ids=ids)['ids'])
        except Exception as e:
            print(f"Error getting existing IDs from cache: {e}")
            existing_ids = set() # Reset to empty set if error occurs

    new_docs, new_ids = [], []
    for id, p in zip(ids, product_cache):
        if id not in existing_ids:
            new_docs.append(p)
            new_ids.append(id)

    if new_ids:
        try:
            chroma_cache_collection.add(
                documents=new_docs,
                ids=new_ids
            )
        except Exception as e:
            print(f"Error adding to cache: {e}")

    sim_flags = identify_similarity_feats(user_query)

    keys = [k for k, v in (sim_flags or {}).items() if v]
    kv = extract_unstructured_feats(user_query, only_keys=keys)

    texts = [user_query]
    if kv:
        kv_str = ", ".join([f"{k}: {v}" for k, v in kv.items()])
        texts.append(kv_str)

    try:
        result = chroma_cache_collection.query(query_texts=texts, n_results=20)
        return result['documents'][0] if result and result['documents'] else []
    except Exception as e:
        print(f"Error querying cache: {e}")
        return []


# --- Flask Routes ---
@app.route('/chat', methods=['POST'])
def chat_endpoint():
    user_query = request.json.get('query')
    top_k = request.json.get('top_k', 20) # Default top_k to 20

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    # Retrieve conversation history and product cache from session
    # Initialize if not present
    conversation_history = session.get('conversation_history', [
        {"role": "system", "content": "You are ZeeVice.ai, an expert on smartphones in the Egyptian market. "
                                      "Be concise, consistent, friendly, and engaging."}
    ])
    products_cache = session.get('products_cache', [])

    # Add user message to history
    conversation_history.append({"role": "user", "content": user_query})

    try:
        # Step 1: Quick intent detection
        intent_prompt = f"""
        Classify this message as one word:
        - 'chitchat' (casual conversation/comparisons/anything but product search)
        - 'product_search' (query about products/specifications)

        User query: {user_query}
        """
        intent_response = openai_agent.chat.completions.create(
            model=LLM,
            messages=[{"role": "system", "content": intent_prompt}]
        )
        intent = intent_response.choices[0].message.content.strip().lower()
        app_rate_limiter.wait_if_needed(intent_response)
        product_flag = False

        bot_reply = ""
        if intent == "chitchat":
            response = openai_agent.chat.completions.create(
                model=LLM,
                messages=conversation_history
            )
            app_rate_limiter.wait_if_needed(response)
            bot_reply = response.choices[0].message.content

        elif intent == "product_search":
            products_data = hybrid_search(user_query, top_k=top_k)
            product_flag = True

            if "error" in products_data:
                bot_reply = f"An error occurred during product search: {products_data['error']}"
            else:
                product_context_lines = []
                for p in products_data["cards"]:
                    line = (
                        f"ID: {p.get('id')} — {p.get('title', 'N/A')} — {p.get('brand', 'N/A')} — {p.get('store', 'N/A')} — {p.get('price', 'N/A')} EGP - "
                        f"{p.get('url', 'N/A')} — {p.get('display_size', 'N/A')} — {p.get('display_type', 'N/A')} — {p.get('ram', 'N/A')} RAM "
                        f"— {p.get('storage', 'N/A')} — {p.get('battery', 'N/A')} — {p.get('cores', 'N/A')} — {p.get('os', 'N/A')} — {p.get('network', 'N/A')}"
                        f"— {p.get('main_camera', 'N/A')} Main Camera — {p.get('front_camera', 'N/A')} Front Camera"
                    )
                    product_context_lines.append(line)

                product_context = "\n".join(product_context_lines)

                # Update products_cache for the session
                products_cache.extend(product_context_lines)
                products_cache = list(set(products_cache)) # Deduplicate

                prompt_cache = cache_builder(products_cache, user_query)

                ppp = f"""Return a list of products ids that u will use to answer this query: {user_query}, in the format:
                [45, 89, 94] 
                from those products:
                {product_context}, {prompt_cache}
                If you don't find any relevant products, return an empty list."""

                ppp_response = openai_agent.chat.completions.create(
                    model=LLM,
                    messages=[{"role": "system", "content": ppp}, {"role": "user", "content": user_query}]
                )
                app_rate_limiter.wait_if_needed(ppp_response)
                product_ids = ppp_response.choices[0].message.content.strip()
                product_ids = product_ids[1:-1].split(",") if product_ids else []

                relevant_products = []
                for pid in product_ids:
                    pid = pid.strip()
                    if pid.isdigit():
                        for p in products_data["cards"]:
                            if str(p.get("id")) == pid:
                                relevant_products.append(p)
                                break
                print(f"Relevant products found: {relevant_products}")

                answer_prompt = f"""
                The user asked: {user_query}
                Here are the top {len(relevant_products)} products matching their needs:
                {relevant_products}
                Provide a clear, concise, and engaging answer.
                Always provide the RELEVANT urls of the product(s) in your answer.
                Don't generate product urls on your own whatsoever.
                Show the specs of each product and why they are good.
                """
                # Append the answer_prompt as a system message for the LLM to consider
                # Note: This adds to the history for the current turn, but we'll manage
                # the actual conversation history separately.
                llm_messages_for_response = conversation_history + [{"role": "system", "content": answer_prompt}]

                response = openai_agent.chat.completions.create(
                    model=LLM,
                    messages=llm_messages_for_response
                )
                app_rate_limiter.wait_if_needed(response)
                bot_reply = response.choices[0].message.content

        else:
            bot_reply = "I'm not sure how to handle that request. Please try rephrasing."

        # Add bot's reply to history
        conversation_history.append({"role": "assistant", "content": bot_reply})

        # Save updated history and cache back to session
        session['conversation_history'] = conversation_history
        session['products_cache'] = products_cache

        if product_flag:
            return jsonify({
                    "response": bot_reply,
                    "search_results": relevant_products,
                    "mentioned_product_ids": product_ids
            })
        else:
            return jsonify({"response": bot_reply})


    except Exception as e:
        print(f"An error occurred: {e}")
        # Add error message to history
        conversation_history.append({"role": "assistant", "content": f"An unexpected error occurred: {e}"})
        session['conversation_history'] = conversation_history
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "ZeeVice AI Backend"})


if __name__ == '__main__':
    print("Starting ZeeVice AI Backend...")
    print("Backend is fully integrated with hybrid search and GraphRAG functionality")

    # By default runs on http://127.0.0.1:5000
    app.run(debug=True, host="0.0.0.0", port=5000)
