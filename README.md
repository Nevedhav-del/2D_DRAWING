⚙️ Engineering Drawing to Structured Assembly System

An automated Vision AI pipeline built with Streamlit, Google Gemini (GenAI SDK), and Pydantic that converts engineering drawings (PNG, JPG, JPEG, PDF) into structured, fully traceable, and queryable assembly data.

🚀 Key Features
Multi-Format Input & Batch Support: Upload up to 4 drawing files incrementally (images or PDFs) with automated sidebar sequence management.

Strict Traceability & Bounding Boxes: Maps every Bill of Materials (BOM) item and callout to exact normalized coordinates ([ymin, xmin, ymax, xmax]) and exact raw text snippets.

Visual Component Inspector: Side-by-side visual source crop viewer allowing users to click any component and inspect its exact visual origin on the drawing.

Component Relationships & Sequence Graph: Extracts inter-connections, fasteners, and step-by-step assembly procedures while distinguishing explicit from inferred steps.

🛡️ Self-Validation Dashboard: Automated grounding auditor and integrity checker that cross-checks extracted outputs against input documents to prevent model hallucinations and verify connection consistency.

Natural Language Query Engine: Ask questions about parts, quantities, dimensions, and drawing views with instant, factually grounded answers.

Structured JSON Export: One-click download of the complete validated assembly schema.

📁 Project Structure

├── app.py              # Main Streamlit user interface (5-tab inspection dashboard)
├── core/
    ├── ai_engine.py    # Gemini Vision API integration with automatic model retry & fallback

🛠️ Installation & Setup
1. Prerequisites
Ensure you have Python 3.10+ installed on your system. If you plan to process PDF engineering drawings, ensure poppler is installed for pdf2image.

2. Clone or Setup Repository
Navigate to your project directory and install the required dependencies:

pip install streamlit google-genai pillow python-dotenv pydantic pdf2image

3. Configure Environment Variables
Create a .env file in the root directory of your project and add your Google Gemini API key:

GEMINI_API_KEY=your_actual_gemini_api_key_here
(Note: You can also input your API key directly via the Streamlit runtime sidebar if not set in the environment).

🖥️ Running the Application
Launch the Streamlit app by running:

streamlit run app.py

Open your browser at http://localhost:8501 to access the interface.

he 5-Tab Inspection Workflow
Step 1: Inspect & Annotate — View the original drawing with automatically overlaid bounding boxes and assembly metadata.

Step 2: Component & BOM Extraction — Review the complete structured BOM table and use the Component Location Inspector to view visual source crops of individual parts.

Step 3: Assembly Relationships & Steps — Examine component inter-connections (fasteners, press-fits, threaded joints) and proposed step-by-step assembly sequences with confidence ratings.

Step 4: Query & Export — Ask custom or preset natural language questions about the drawing and download the clean JSON export.

Step 5: 🛡️ Self-Validation Dashboard — Run automated grounding audits, check for orphaned connections, verify bounding box coverage, and review the full cross-check matrix.
    └── models.py       # Pydantic data schemas (AssemblyData, Part, Connection, AssemblyStep)
├── requirements.txt    # Python dependencies
└── .env                # Environment configuration (GEMINI_API_KEY)
