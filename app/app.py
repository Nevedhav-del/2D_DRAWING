import os
import streamlit as st
from PIL import Image, ImageDraw
from dotenv import load_dotenv
import uuid
from datetime import datetime
import json

load_dotenv()

from core.models import AssemblyData, Part
from core.ai_engine import extract_assembly_from_image, dynamic_verify_extraction, query_assembly_data

Image.MAX_IMAGE_PIXELS = None

# Robust PDF conversion using PyMuPDF (fitz) - eliminates Poppler binary dependency errors entirely
PYMUPDF_SUPPORT = False
try:
    import fitz  # PyMuPDF
    PYMUPDF_SUPPORT = True
except ImportError:
    pass

st.set_page_config(page_title="Engineering Drawing Analyzer", layout="wide", page_icon="⚙️")

st.title("⚙️ Engineering Drawing to Structured Assembly System")
st.markdown("Automated Vision AI pipeline for engineering drawings. Extracts BOM, callouts, mechanical connections, sequence graph, **Full Traceability**, **Self-Validation Dashboard**, **Dynamic Independent Verification**, and **Reference Accuracy Benchmarking**.")

st.sidebar.header("")
env_has_key = bool(os.environ.get("GEMINI_API_KEY"))
if env_has_key:
    st.sidebar.success("")
else:
    st.sidebar.warning("⚠️ GEMINI_API_KEY not found in environment. Please ensure your .env file is configured.")

st.sidebar.header("📁 Document Source & Incremental Batch")

uploaded_files = st.sidebar.file_uploader(
    "Upload Drawing Images or PDFs (PNG, JPG, JPEG, PDF - Up to 4 files)", 
    type=["png", "jpg", "jpeg", "pdf"],
    accept_multiple_files=True
)

if uploaded_files and len(uploaded_files) > 4:
    st.sidebar.warning("⚠️ More than 4 files uploaded. Processing will be limited to the first 4 files.")
    uploaded_files = uploaded_files[:4]

if "current_data" not in st.session_state:
    st.session_state.current_data = None
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "source_name" not in st.session_state:
    st.session_state.source_name = None
if "batch_results" not in st.session_state:
    st.session_state.batch_results = {}
if "user_corrections" not in st.session_state:
    st.session_state.user_corrections = {}
if "reference_benchmarks" not in st.session_state:
    st.session_state.reference_benchmarks = {}

# Strict Programmatic Rule-Based Component Scoring and Grounding Audit Engine
def evaluate_part_confidence_and_status(part: Part, all_bom_items: list):
    checks = []
    
    # 1. Programmatic Spatial Grounding Audit
    has_valid_box = bool(part.bounding_box and len(part.bounding_box) == 4 and 
                         0.0 <= part.bounding_box[0] < part.bounding_box[2] <= 1.0 and 
                         0.0 <= part.bounding_box[1] < part.bounding_box[3] <= 1.0)
    checks.append({
        "rule_name": "Spatial Grounding Check",
        "weight": 0.30,
        "passed": has_valid_box,
        "detail": "Bounding box coordinates are programmatically valid and within image boundaries." if has_valid_box else "Missing, malformed, or out-of-bounds spatial bounding box."
    })

    # 2. Programmatic Text Snippet Traceability Audit
    has_snippet = bool(part.extracted_text_snippet and len(part.extracted_text_snippet.strip()) > 0)
    checks.append({
        "rule_name": "Text Snippet Traceability",
        "weight": 0.25,
        "passed": has_snippet,
        "detail": f"Raw snippet verified: '{part.extracted_text_snippet}'" if has_snippet else "Missing raw text snippet reference."
    })

    # 3. Programmatic BOM Quantity Validity Audit
    is_qty_valid = str(part.qty).strip().isdigit() and int(str(part.qty)) > 0
    checks.append({
        "rule_name": "BOM Quantity Validity",
        "weight": 0.25,
        "passed": is_qty_valid,
        "detail": f"Quantity '{part.qty}' is a valid positive integer." if is_qty_valid else f"Invalid quantity value '{part.qty}'."
    })

    # 4. Programmatic Item-to-Callout Uniqueness Mapping Audit
    is_item_valid = isinstance(part.item, int) and part.item > 0 and all_bom_items.count(part.item) == 1
    checks.append({
        "rule_name": "Item-to-Callout Mapping Integrity",
        "weight": 0.20,
        "passed": is_item_valid,
        "detail": f"Item number {part.item} is uniquely mapped." if is_item_valid else f"Item number {part.item} is missing, invalid, or duplicated."
    })

    total_score = sum(c["weight"] for c in checks if c["passed"])
    has_critical_failure = not is_qty_valid or not is_item_valid

    if has_critical_failure:
        status = "Failed Validation ❌"
    elif total_score >= 0.80 and not any(not c["passed"] for c in checks if c["rule_name"] in ["Spatial Grounding Check", "Text Snippet Traceability"]):
        status = "Validated 🟢"
    elif total_score >= 0.50:
        status = "Needs Review 🟠"
    else:
        status = "Failed Validation ❌"

    part.confidence_score = round(total_score, 2)
    part.validation_status = status
    part.check_details = checks
    return part

def run_validation_checks(data: AssemblyData):
    checks = []
    bom_items = [p.item for p in data.parts]
    bom_item_set = set(bom_items)
    
    for p in data.parts:
        evaluate_part_confidence_and_status(p, bom_items)

    broken_conns = []
    for c in data.connections:
        endpoints_valid = (c.part_a_item in bom_item_set) and (c.part_b_item in bom_item_set)
        fastener_valid = True
        if c.fastener_item is not None:
            fastener_valid = c.fastener_item in bom_item_set
            
        if not endpoints_valid or not fastener_valid or not c.drawing_evidence:
            broken_conns.append(c)

    checks.append({
        "rule_name": "Connection Integrity & Fastener Rule",
        "result": "Passed" if len(broken_conns) == 0 else "Failed",
        "evidence_used": f"BOM Item Numbers ({list(bom_item_set)}) vs Connection Endpoints & Fasteners",
        "reason": "All connections reference valid BOM parts/fasteners with explicit drawing evidence." if len(broken_conns) == 0 else f"Found {len(broken_conns)} connection(s) with invalid item references or missing drawing evidence."
    })

    unverified_steps = [s for s in data.assembly_steps if s.confidence.lower() != "explicit" or not s.evidence]
    checks.append({
        "rule_name": "Assembly Sequence Verification Rule",
        "result": "Passed" if len(unverified_steps) == 0 else "Requires Review",
        "evidence_used": "Assembly Step Confidence Level ('Explicit' vs 'Inferred') & Supporting View/Note",
        "reason": "All assembly steps are explicitly verified by drawing notes/views." if len(unverified_steps) == 0 else f"{len(unverified_steps)} step(s) are inferred and flagged as requiring manual review."
    })

    missing_boxes = [p.item for p in data.parts if not p.bounding_box]
    checks.append({
        "rule_name": "Spatial Grounding Rule",
        "result": "Passed" if len(missing_boxes) == 0 else "Requires Review",
        "evidence_used": "Normalized Bounding Box Coordinates ([ymin, xmin, ymax, xmax])",
        "reason": "All parts possess spatial coordinate boxes." if len(missing_boxes) == 0 else f"Items {missing_boxes} lack bounding box coordinates."
    })

    invalid_qtys = [p.item for p in data.parts if not str(p.qty).strip().isdigit() or int(str(p.qty)) <= 0]
    checks.append({
        "rule_name": "BOM Quantity Validity Rule",
        "result": "Passed" if len(invalid_qtys) == 0 else "Failed",
        "evidence_used": "Extracted Part Quantity Field vs BOM Row Expectations",
        "reason": "All part quantities are valid positive integers." if len(invalid_qtys) == 0 else f"Items {invalid_qtys} have invalid quantities."
    })

    return checks

if uploaded_files:
    active_key = os.environ.get("GEMINI_API_KEY")
    
    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        file_name = uploaded_file.name
        
        if file_name not in st.session_state.batch_results:
            st.info(f"⏳ Incrementally processing new file ({idx}/{len(uploaded_files)}): **{file_name}**...")
            
            try:
                extracted_data = None
                active_image = None
                
                if file_name.lower().endswith(".pdf"):
                    pdf_bytes = uploaded_file.read()
                    pages_images = []
                    
                    if PYMUPDF_SUPPORT:
                        try:
                            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                            for page_num in range(len(doc)):
                                page = doc[page_num]
                                pix = page.get_pixmap(dpi=200)
                                img_path_temp = f"_temp_page_{page_num}.png"
                                pix.save(img_path_temp)
                                pages_images.append(Image.open(img_path_temp).convert("RGB"))
                        except Exception as e_fitz:
                            st.error(f"❌ PyMuPDF rendering error for `{file_name}`: {e_fitz}")
                    else:
                        st.error(f"❌ PyMuPDF (`pymupdf`) is required to process PDFs without Poppler. Please install it via `pip install pymupdf`.")
                        continue
                            
                    if not pages_images:
                        st.error(f"❌ Could not render pages for PDF `{file_name}`.")
                        continue
                        
                    # Process EVERY relevant PDF page and combine results
                    active_image = pages_images[0]
                    extracted_data = extract_assembly_from_image(
                        active_image, file_name, page_number=1, api_key=active_key
                    )
                    
                    if len(pages_images) > 1:
                        for page_num, page_img in enumerate(pages_images[1:], start=2):
                            rgb_page = page_img.convert("RGB")
                            page_data = extract_assembly_from_image(
                                rgb_page, file_name, page_number=page_num, api_key=active_key
                            )
                            for p in page_data.parts:
                                p.extracted_text_snippet = f"[Page {page_num}] {p.extracted_text_snippet or ''}"
                            extracted_data.parts.extend(page_data.parts)
                            extracted_data.connections.extend(page_data.connections)
                            extracted_data.assembly_steps.extend(page_data.assembly_steps)
                else:
                    active_image = Image.open(uploaded_file).convert("RGB")
                    extracted_data = extract_assembly_from_image(
                        active_image, file_name, page_number=1, api_key=active_key
                    )
                
                bom_list = [p.item for p in extracted_data.parts]
                for p in extracted_data.parts:
                    evaluate_part_confidence_and_status(p, bom_list)

                st.session_state.batch_results[file_name] = {
                    "data": extracted_data,
                    "image": active_image,
                    "pipeline_metadata": {
                        "run_id": str(uuid.uuid4()),
                        "pipeline_version": "v2.9.0-fully-dynamic",
                        "ruleset_version": "2026.2",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
                st.success(f"✅ Successfully processed and audited: {file_name}")
                
            except Exception as e:
                st.error(f"❌ Failed to process {file_name}: {e}")

    if st.session_state.current_data is None and st.session_state.batch_results:
        first_key = list(st.session_state.batch_results.keys())[0]
        st.session_state.current_data = st.session_state.batch_results[first_key]["data"]
        st.session_state.current_image = st.session_state.batch_results[first_key]["image"]
        st.session_state.source_name = first_key

if st.session_state.batch_results:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📂 Processed Files Sequence")
    
    batch_keys = list(st.session_state.batch_results.keys())
    current_index = batch_keys.index(st.session_state.source_name) if st.session_state.source_name in batch_keys else 0
    
    selected_batch_file = st.sidebar.selectbox(
        "Select file to view output tabs:", 
        batch_keys,
        index=current_index
    )
    
    if selected_batch_file != st.session_state.source_name:
        st.session_state.current_data = st.session_state.batch_results[selected_batch_file]["data"]
        st.session_state.current_image = st.session_state.batch_results[selected_batch_file]["image"]
        st.session_state.source_name = selected_batch_file

if st.session_state.current_data is None:
    st.info("👈 Please upload any engineering drawing image or PDF in the sidebar to begin dynamic processing and unlock all 6 inspection tabs.")
else:
    current_data: AssemblyData = st.session_state.current_data
    image: Image.Image = st.session_state.current_image
    source_name = st.session_state.source_name

    bom_item_list = [p.item for p in current_data.parts]
    for p in current_data.parts:
        evaluate_part_confidence_and_status(p, bom_item_list)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Step 1: Inspect & Annotate", 
        "Step 2: Component & BOM Extraction", 
        "Step 3: Assembly Relationships & Steps", 
        "Step 4: Query & Export",
        "Step 5: 🛡️ Self-Validation Dashboard",
        "Step 6: 📏 Reference Benchmarking"
    ])

    with tab1:
        st.subheader(f"Assembly View: {current_data.name} (File: {source_name})")
        c1, c2 = st.columns([2, 1])

        width, height = image.size
        annotated_img = image.copy()
        draw = ImageDraw.Draw(annotated_img)

        for part in current_data.parts:
            if part.bounding_box and len(part.bounding_box) == 4:
                ymin, xmin, ymax, xmax = part.bounding_box
                if xmax <= 1.0 and ymax <= 1.0 and xmin >= 0.0 and ymin >= 0.0:
                    rx1, ry1, rx2, ry2 = xmin * width, ymin * height, xmax * width, ymax * height
                else:
                    rx1, ry1, rx2, ry2 = xmin, ymin, xmax, ymax
                
                rx1 = max(0, min(rx1, width))
                ry1 = max(0, min(ry1, height))
                rx2 = max(rx1, min(rx2, width))
                ry2 = max(ry1, min(ry2, height))

                draw.rectangle([rx1, ry1, rx2, ry2], outline="red", width=3)

        with c1:
            st.image(
                annotated_img, 
                caption=f"Drawing: {source_name} (Programmatically Grounded Bounding Boxes Outlined)", 
                use_container_width=True
            )

        with c2:
            st.markdown("### Drawing Metadata & Uncertainty Log")
            st.write(f"**Extracted Assembly Title:** {current_data.name}")
            st.write(f"**Material:** {current_data.material or 'Not specified'}")
            st.write(f"**Scale:** {current_data.scale or 'N/A'}")
            st.write(f"**Detected Views:** {', '.join(current_data.detected_views) if current_data.detected_views else 'General View'}")
            
            st.markdown("---")
            st.markdown("#### ⚠️ Programmatically Flagged Uncertainties:")
            if hasattr(current_data, "unresolved_uncertainties") and current_data.unresolved_uncertainties:
                for unc in current_data.unresolved_uncertainties:
                    st.warning(f"- {unc}")
            else:
                st.info("ℹ️ No unverified ambiguities reported after programmatic spatial and BOM audits.")

    with tab2:
        st.subheader("Bill of Materials (BOM) with Traceability & Scoring Status")
        if current_data.parts:
            table_rows = []
            for p in current_data.parts:
                table_rows.append({
                    "Item #": p.item,
                    "Part Name": p.name,
                    "Qty": p.qty,
                    "Page": p.page_number,
                    "Confidence Score": f"{p.confidence_score:.2f}" if p.confidence_score is not None else "N/A",
                    "Validation Status": p.validation_status or "Pending",
                    "Dimensions": p.dimensions or "-",
                    "Thread Spec": p.thread_spec or "-",
                    "Material": p.material or "-"
                })
            st.table(table_rows)

            st.markdown("---")
            st.subheader("Component Inspector, Score Breakdown & Manual Correction")
            part_names = [f"Item {p.item}: {p.name} ({p.validation_status})" for p in current_data.parts]
            selected_part_str = st.selectbox("Select component to inspect or correct:", part_names, key=f"part_select_{source_name}")
            
            selected_item_num = int(selected_part_str.split(":")[0].replace("Item ", ""))
            selected_part = next(p for p in current_data.parts if p.item == selected_item_num)

            col_detail, col_crop = st.columns(2)
            with col_detail:
                st.markdown(f"#### Traceability & Score Breakdown: {selected_part.name}")
                st.write(f"**Item Number:** {selected_part.item}")
                st.write(f"**Quantity:** {selected_part.qty}")
                st.write(f"**Source Page:** Page {selected_part.page_number}")
                st.write(f"**Calculated Score:** `{selected_part.confidence_score}`")
                st.write(f"**Assigned Status:** `{selected_part.validation_status}`")
                st.write(f"**Raw Text Snippet:** `{selected_part.extracted_text_snippet or 'None'}`")
                
                st.markdown("##### Programmatic Validation Checks:")
                if selected_part.check_details:
                    for chk in selected_part.check_details:
                        icon = "🟢 Passed" if chk["passed"] else "❌ Failed"
                        st.markdown(f"- **{chk['rule_name']}** (Weight: {chk['weight']}): {icon} — *{chk['detail']}*")

                new_item_num = st.number_input("Correct Item #:", value=int(selected_part.item), step=1, key=f"corr_item_{selected_item_num}_{source_name}")
                new_name_corr = st.text_input("Correct Part Name:", value=selected_part.name, key=f"corr_name_{selected_item_num}_{source_name}")
                new_qty_corr = st.number_input("Correct Quantity:", value=int(selected_part.qty) if str(selected_part.qty).isdigit() else 1, key=f"corr_qty_{selected_item_num}_{source_name}")
                
                if st.button("Save Correction & Re-calculate Scores", key=f"save_corr_{selected_item_num}_{source_name}"):
                    correction_key = f"{source_name}_item_{selected_item_num}"
                    st.session_state.user_corrections[correction_key] = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "item": selected_item_num,
                        "original_values": {"item": selected_part.item, "name": selected_part.name, "qty": selected_part.qty},
                        "corrected_values": {"item": new_item_num, "name": new_name_corr, "qty": new_qty_corr},
                        "review_outcome": "Re-scored & Updated 🟢"
                    }
                    selected_part.item = int(new_item_num)
                    selected_part.name = new_name_corr
                    selected_part.qty = new_qty_corr
                    
                    updated_bom_list = [p.item for p in current_data.parts]
                    evaluate_part_confidence_and_status(selected_part, updated_bom_list)

                    st.success(f"✅ Successfully recorded correction for Item {selected_item_num}! Programmatic scores re-calculated.")
                    st.rerun()

            with col_crop:
                if selected_part.bounding_box and len(selected_part.bounding_box) == 4:
                    ymin, xmin, ymax, xmax = selected_part.bounding_box
                    if xmax <= 1.0 and ymax <= 1.0 and xmin >= 0.0 and ymin >= 0.0:
                        box_xmin = xmin * width
                        box_ymin = ymin * height
                        box_xmax = xmax * width
                        box_ymax = ymax * height
                    else:
                        box_xmin, box_ymin, box_xmax, box_ymax = xmin, ymin, xmax, ymax
                    
                    box_xmin = max(0, min(box_xmin, width))
                    box_ymin = max(0, min(box_ymin, height))
                    box_xmax = max(box_xmin, min(box_xmax, width))
                    box_ymax = max(box_ymin, min(box_ymax, height))
                    
                    if box_xmax > box_xmin and box_ymax > box_ymin:
                        cropped_img = image.crop((box_xmin, box_ymin, box_xmax, box_ymax))
                        st.image(cropped_img, caption=f"Visual Proof: Source Crop for Item {selected_part.item}", width=280)
                    else:
                        st.info("Invalid bounding box dimensions for cropping.")
                else:
                    st.info("No specific bounding box coordinates identified for this callout.")
        else:
            st.warning("No individual parts detected in the drawing BOM.")

    with tab3:
        col_rel, col_seq = st.columns(2)

        with col_rel:
            st.subheader("Component Inter-connections & Source Proof")
            if current_data.connections:
                bom_items = {p.item for p in current_data.parts}
                for idx, conn in enumerate(current_data.connections, 1):
                    part_a = next((p for p in current_data.parts if p.item == conn.part_a_item), None)
                    part_b = next((p for p in current_data.parts if p.item == conn.part_b_item), None)
                    
                    part_a_name = part_a.name if part_a else f"Item {conn.part_a_item}"
                    part_b_name = part_b.name if part_b else f"Item {conn.part_b_item}"
                    
                    valid_endpoints = (conn.part_a_item in bom_items) and (conn.part_b_item in bom_items)
                    valid_fastener = True if conn.fastener_item is None else (conn.fastener_item in bom_items)
                    
                    fastener_str = f" via Fastener Item {conn.fastener_item}" if conn.fastener_item else ""
                    status_icon = "🟢 Verified" if (valid_endpoints and valid_fastener and conn.drawing_evidence) else "❌ Unverified / Orphaned"
                    
                    st.markdown(f"**{idx}. {part_a_name} ↔ {part_b_name}** ({status_icon})")
                    st.write(f"- **Type:** {conn.connection_type}{fastener_str}")
                    st.write(f"- **Source File / Page:** `{source_name}` (Page {conn.page_number})")
                    st.write(f"- **Drawing Evidence / View:** {conn.drawing_evidence or 'None specified'}")
                    if conn.source_snippet:
                        st.markdown(f"- **Source Text Snippet:** `{conn.source_snippet}`")
                    st.divider()
            else:
                st.info("No specific connection interfaces identified.")

        with col_seq:
            st.subheader("Proposed Assembly Sequence & References")
            if current_data.assembly_steps:
                for step in current_data.assembly_steps:
                    is_explicit = step.confidence.lower() == "explicit"
                    conf_color = "🟢 Explicit" if is_explicit else "🟠 Inferred (Requires Review)"
                    with st.expander(f"Step {step.step}: {step.action} [{conf_color}]"):
                        st.write(f"**Confidence Level:** {step.confidence.upper()}")
                        st.write(f"**Involved Items:** {step.parts}")
                        st.write(f"**Supporting View / Evidence:** {step.evidence or 'Not established by drawing'}")
                        if not is_explicit:
                            st.warning("⚠️ **Notice:** This step is inferred and does not constitute a verified manufacturing instruction until manually reviewed.")
                        if step.source_reference:
                            st.write(f"**Source Reference:** {step.source_reference}")
            else:
                st.info("No assembly steps could be determined.")

    with tab4:
        st.subheader("💬 Natural Language Retrieval & Query Engine")
        
        sample_queries = [
            "What are the parts in this assembly?",
            "What is the quantity of each part?",
            "Which parts are connected using a bolt or threads?",
            "What is the likely assembly sequence?",
            "Which drawing view and text snippets support this information?"
        ]
        
        selected_q = st.selectbox("Select a sample query or write your own below:", ["-- Custom Query --"] + sample_queries, key=f"q_select_{source_name}")
        user_query = st.text_input("Enter your question about the assembly:", value="" if selected_q == "-- Custom Query --" else selected_q, key=f"q_input_{source_name}")

        if st.button("Query Drawing Data", key=f"q_btn_{source_name}") and user_query:
            with st.spinner("Retrieving programmatically audited facts with visual evidence..."):
                active_key = os.environ.get("GEMINI_API_KEY")
                answer = query_assembly_data(user_query, current_data, api_key=active_key)
                st.markdown("### Answer with Programmatic Source Citations:")
                st.markdown(answer)

        st.divider()
        st.subheader("📥 Structured JSON Export with Scoring & Validation Metadata")
        
        meta = st.session_state.batch_results.get(source_name, {}).get("pipeline_metadata", {})
        validation_checks_list = run_validation_checks(current_data)
        failed_checks = [c for c in validation_checks_list if c["result"] == "Failed"]

        export_payload = {
            "pipeline_metadata": meta,
            "source_file": source_name,
            "assembly_data": current_data.model_dump(),
            "validation_results": {
                "total_parts": len(current_data.parts),
                "checks_performed": validation_checks_list,
                "review_status": "Passed" if len(failed_checks) == 0 else "Requires Review",
                "failure_reasons": [c["reason"] for c in validation_checks_list if c["result"] != "Passed"]
            },
            "user_corrections": st.session_state.user_corrections
        }

        json_output = json.dumps(export_payload, indent=2)
        
        st.download_button(
            label="Download Structured JSON Export",
            data=json_output,
            file_name=f"{current_data.name.lower().replace(' ', '_').replace('[', '').replace(']', '')}_scored_assembly.json",
            mime="application/json"
        )
        
        with st.expander("View Extended JSON Schema"):
            st.code(json_output, language="json")

    with tab5:
        st.subheader("🛡️ Self-Validation Dashboard & Dynamic Auditor")
        st.markdown("Evaluates programmatic confidence scores, weighted rule checks, and status assignments (`Validated`, `Needs Review`, `Failed Validation`) dynamically for any uploaded drawing.")

        validation_checks_list = run_validation_checks(current_data)

        validated_count = sum(1 for p in current_data.parts if p.validation_status and "Validated" in p.validation_status)
        review_count = sum(1 for p in current_data.parts if p.validation_status and "Needs Review" in p.validation_status)
        failed_count = sum(1 for p in current_data.parts if p.validation_status and "Failed" in p.validation_status)

        col_v1, col_v2, col_v3 = st.columns(3)
        with col_v1:
            st.metric(label="Validated Parts", value=f"{validated_count} / {len(current_data.parts)}")
        with col_v2:
            st.metric(label="Needs Review Parts", value=review_count)
        with col_v3:
            st.metric(label="Failed Validation Parts", value=failed_count)

        st.markdown("---")
        st.markdown("### 📊 Component Score & Rule Breakdown Table")
        
        comp_score_rows = []
        for p in current_data.parts:
            failed_rules = [c["rule_name"] for c in p.check_details if not c["passed"]]
            comp_score_rows.append({
                "Item #": p.item,
                "Part Name": p.name,
                "Score": f"{p.confidence_score:.2f}" if p.confidence_score is not None else "N/A",
                "Status": p.validation_status or "Pending",
                "Failed Checks": ", ".join(failed_rules) if failed_rules else "None (All Passed)"
            })
        st.table(comp_score_rows)

        st.markdown("---")
        st.markdown("### 📋 Rule-Based Source Evidence Validation Log")
        rule_table_rows = []
        for chk in validation_checks_list:
            status_badge = "Passed 🟢" if chk["result"] == "Passed" else ("Failed ❌" if chk["result"] == "Failed" else "Requires Review 🟠")
            rule_table_rows.append({
                "Rule Name": chk["rule_name"],
                "Validation Status": status_badge,
                "Source Evidence Used": chk["evidence_used"],
                "Reason / Failure Detail": chk["reason"]
            })
        st.table(rule_table_rows)

        st.markdown("---")
        st.markdown("### 🔍 Dynamic Independent Verification Audit")
        st.markdown("Runs an independent dynamic cross-check comparing extracted entities directly against visual features in the source image.")
        
        if st.button("Run Dynamic Image Audit", key=f"dynamic_audit_{source_name}"):
            with st.spinner("Performing dynamic visual audit..."):
                active_key = os.environ.get("GEMINI_API_KEY")
                audit_report = dynamic_verify_extraction(image, current_data.model_dump_json(), api_key=active_key)
                st.markdown(audit_report)

        if st.session_state.user_corrections:
            st.markdown("---")
            st.markdown("### 📝 User Correction Audit Trail")
            corr_rows = []
            for k, corr in st.session_state.user_corrections.items():
                corr_rows.append({
                    "Timestamp": corr["timestamp"],
                    "Item #": corr["item"],
                    "Original Values": str(corr["original_values"]),
                    "Corrected Values": str(corr["corrected_values"]),
                    "Review Outcome": corr["review_outcome"]
                })
            st.table(corr_rows)

    with tab6:
        st.subheader("📏 Reference Benchmarking & Accuracy Evaluation")
        st.markdown("Compare the extracted assembly data against a manually verified ground-truth reference BOM and connection list to measure accuracy.")
        
        with st.form(key=f"benchmark_form_{source_name}"):
            st.markdown("#### Enter Ground-Truth Reference BOM (JSON Format)")
            default_ref_json = json.dumps([
                {"item": 1, "name": "Base Plate", "qty": 1},
                {"item": 2, "name": "Bearing Housing", "qty": 2}
            ], indent=2)
            
            ref_json_input = st.text_area("Expected Reference BOM Items:", value=default_ref_json, height=150)
            run_benchmark_btn = st.form_submit_button("Run Accuracy Benchmark")
            
            if run_benchmark_btn:
                try:
                    ref_list = json.loads(ref_json_input)
                    extracted_parts = current_data.parts
                    
                    matched_count = 0
                    mismatched_items = []
                    
                    extracted_map = {p.item: (p.name, p.qty) for p in extracted_parts}
                    
                    for ref in ref_list:
                        r_item = ref.get("item")
                        r_name = ref.get("name").strip().lower()
                        r_qty = ref.get("qty")
                        
                        if r_item in extracted_map:
                            ex_name, ex_qty = extracted_map[r_item]
                            if ex_name.strip().lower() == r_name and int(ex_qty) == int(r_qty):
                                matched_count += 1
                            else:
                                mismatched_items.append(f"Item {r_item}: Expected ({ref.get('name')}, Qty {r_qty}) vs Extracted ({ex_name}, Qty {ex_qty})")
                        else:
                            mismatched_items.append(f"Item {r_item} missing from extraction.")
                            
                    total_ref = len(ref_list)
                    accuracy_pct = (matched_count / total_ref) * 100 if total_ref > 0 else 0.0
                    
                    st.metric(label="Reference BOM Accuracy", value=f"{accuracy_pct:.1f}%", delta=f"{matched_count} / {total_ref} matched perfectly")
                    
                    if mismatched_items:
                        st.warning("⚠️ Discrepancies found against reference ground truth:")
                        for m in mismatched_items:
                            st.write(f"- {m}")
                    else:
                        st.success("🎉 Extracted BOM matches ground-truth reference with 100% precision!")
                        
                except Exception as e:
                    st.error(f"❌ Invalid reference JSON format: {e}")
