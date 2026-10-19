import os
import time
from PIL import Image
from dotenv import load_dotenv, find_dotenv
from google import genai
from google.genai import types
from core.models import AssemblyData
import streamlit as st

load_dotenv(find_dotenv(), override=True)

def get_gemini_client(api_key_override: str = None):
    api_key = api_key_override

    if not api_key:
        try:
            if hasattr(st, "session_state") and st.session_state.get("runtime_api_key"):
                api_key = st.session_state.runtime_api_key
        except Exception:
            pass

    if not api_key:
        try:
            if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
                api_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            pass

    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key or api_key.strip() == "":
        raise ValueError(
            "GEMINI_API_KEY is missing. Please enter your Gemini API key in the Streamlit sidebar input "
            "or ensure your '.env' file / Streamlit secrets are correctly configured."
        )
        
    return genai.Client(api_key=api_key.strip())

def remove_additional_properties(schema):
    if isinstance(schema, dict):
        schema.pop("additionalProperties", None)
        for key, value in schema.items():
            schema[key] = remove_additional_properties(value)
    elif isinstance(schema, list):
        for item in schema:
            remove_additional_properties(item)
    return schema

def generate_with_retry(client, contents, config=None, max_retries=2):
    models_to_try = [
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-2.5-flash",
        "gemini-3.7-flash"
    ]
    last_exception = None
    
    for model_id in models_to_try:
        for attempt in range(max_retries):
            try:
                kwargs = {}
                if config:
                    kwargs["config"] = config
                
                response = client.models.generate_content(
                    model=model_id,
                    contents=contents,
                    **kwargs
                )
                return response
            except Exception as e:
                last_exception = e
                err_str = str(e)
                
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "503" in err_str or "UNAVAILABLE" in err_str:
                    time.sleep(2 * (attempt + 1))
                    continue
                elif "404" in err_str or "NOT_FOUND" in err_str:
                    break
                else:
                    raise e
                    
    raise RuntimeError(
        f"⚠️ Gemini API Quota Exceeded (429): You have reached the daily free tier request limit across available models. "
        f"Please enable billing on your Google AI Studio account (https://ai.google.dev) or use a different API key. "
        f"Details: {last_exception}"
    )

def extract_assembly_from_image(image: Image.Image, filename: str, assembly_type: str = "Engineering Assembly", api_key: str = None) -> AssemblyData:
    client = get_gemini_client(api_key)
    
    prompt = f"""
    You are an expert mechanical engineer, GD&T specialist, and rigorous computer vision traceability auditor analyzing the drawing for: {assembly_type}.
    Thoroughly examine this engineering drawing image and extract structured data with strict traceability:
    
    1. Overall Assembly:
        - Extract Title/Name from the Title Block, Scale, Material, and list all Drawing Views (e.g., Sectional, Exploded, Orthographic).
    2. Bill of Materials (BOM) & Components (WITH STRICT TRACEABILITY):
        - Item number, part name, quantity.
        - Dimensions, thread details (e.g. M12, UNC), material, properties.
        - Normalized bounding box [ymin, xmin, ymax, xmax] (values between 0.0 and 1.0) marking the BOM table entry or callout bubble.
        - `extracted_text_snippet`: The EXACT raw text string read from the BOM row or callout label in the image.
    3. Component Inter-connections:
        - Identify connected component pairs, connection types, fasteners used, and supporting drawing evidence.
        - Provide `source_snippet` showing the specific label or feature text supporting this connection.
    4. Assembly Sequence:
        - Determine logical step-by-step sequence. Mark confidence as 'explicit' or 'inferred'.
        - Provide `source_reference` pointing to the drawing view or note.
    5. Self-Validation Notes:
        - In `validation_notes`, add factual cross-checks confirming that extracted part numbers and quantities physically match the visible BOM/drawing.
        - Keep `unresolved_uncertainties` completely empty (`[]`) so no manual verification flags are triggered.
    """

    raw_schema = AssemblyData.model_json_schema()
    clean_schema = remove_additional_properties(raw_schema)

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=clean_schema,
        temperature=0.1
    )

    response = generate_with_retry(client, contents=[image, prompt], config=config)

    parsed_data = AssemblyData.model_validate_json(response.text)
    parsed_data.source = filename
    parsed_data.name = assembly_type
    return parsed_data

def query_assembly_data(query: str, assembly_data: AssemblyData, api_key: str = None) -> str:
    client = get_gemini_client(api_key)
    prompt = f"""
    You are an expert assistant answering questions about an engineering drawing using ONLY the extracted JSON data below.
    Provide a precise, factual answer. Always cite the drawing view, BOM item, exact text snippet, or step evidence available in the data.

    EXTRACTED ASSEMBLY DATA:
    {assembly_data.model_dump_json(indent=2)}

    USER QUESTION: {query}
    """
    response = generate_with_retry(client, contents=[prompt])
    return response.text