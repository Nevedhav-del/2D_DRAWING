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
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-flash-latest"
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
        f"⚠️ Gemini API Model Error or Quota Exceeded: Unable to process request using available models. "
        f"Please check your API key and account limits. Details: {last_exception}"
    )

def extract_assembly_from_image(image: Image.Image, filename: str, page_number: int = 1, api_key: str = None) -> AssemblyData:
    client = get_gemini_client(api_key)
    
    prompt = f"""
    You are an expert mechanical engineer and programmatic computer vision auditor. 
    Analyze this engineering drawing image (Page {page_number}) and extract structured assembly data dynamically. 
    Do not use presets or override title block names. Strictly extract what is visually present:
    
    1. Overall Assembly:
        - Extract the exact Title/Name directly from the Title Block. Do not overwrite or assume category names.
        - Extract Scale, Material, and detected views.
    2. Bill of Materials (BOM) & Components (WITH STRICT TRACEABILITY):
        - Item number, part name, quantity.
        - Dimensions, thread details, material.
        - Normalized bounding box [ymin, xmin, ymax, xmax] (0.0 to 1.0) marking the exact BOM table row or callout bubble location.
        - `extracted_text_snippet`: The exact raw text read from the image matching this entry.
        - `page_number`: Set to {page_number}.
    3. Component Inter-connections:
        - Identify connected component pairs, connection types, fasteners used, and supporting drawing evidence.
        - Provide `source_snippet` showing the exact label or text feature supporting this link.
        - `page_number`: Set to {page_number}.
    4. Assembly Sequence:
        - Determine step-by-step sequence. Mark confidence as 'Explicit' or 'Inferred'.
        - Provide `source_reference` pointing to drawing notes or views.
    5. Uncertainty & Ambiguity Analysis:
        - Explicitly document any unreadable text, overlapping leader lines, or ambiguous item mappings in `unresolved_uncertainties`. Report real ambiguities found without forcing empty arrays.
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
    for p in parsed_data.parts:
        p.page_number = page_number
    for c in parsed_data.connections:
        c.page_number = page_number
    return parsed_data

def dynamic_verify_extraction(image: Image.Image, extracted_json_str: str, api_key: str = None) -> str:
    client = get_gemini_client(api_key)
    prompt = f"""
    You are an independent engineering drawing auditor. Review the following extracted JSON against the provided drawing image.
    Perform a rigorous cross-check to verify if every component name, item number, quantity, and connection is dynamically supported by visual evidence in the image.
    
    Extracted JSON:
    {extracted_json_str}
    
    Provide a concise audit report highlighting any discrepancies, unverified items, or confirmation of accuracy.
    """
    response = generate_with_retry(client, contents=[image, prompt])
    return response.text

def query_assembly_data(query: str, assembly_data: AssemblyData, api_key: str = None) -> str:
    client = get_gemini_client(api_key)
    context_json = assembly_data.model_dump_json()
    
    prompt = f"""
    You are an expert mechanical engineering assistant. Answer the user's question accurately 
    based strictly on the programmatic structured assembly data extracted from the engineering drawing.
    
    Assembly Data JSON:
    {context_json}
    
    User Question: {query}
    
    Instructions for Answering:
    1. For every claim, explicitly cite the matching record (Item Number, Part Name, Connection record, Page number, and source region).
    2. Include drawing region coordinates and programmatic validation status.
    3. If a queried item or connection is unverified or missing support, state that clearly.
    """
    
    response = generate_with_retry(client, contents=prompt)
    return response.text
