# Document Extraction & PDF Analysis

This project provides an advanced, automated pipeline for analyzing, OCR-processing, and highlighting specialized documents such as Mechanical Floor Plans. It utilizes a combination of machine learning pipelines, Optical Character Recognition (OCR), and rule-based text matching to intelligently process scanned and digital PDFs.

## Architecture & Model Pipeline

The extraction engine relies on several state-of-the-art ML models to understand document scope, extract nested details, and classify page layouts.

### 1. Optical Character Recognition (OCR)
- **Model:** `EasyOCR` (English)
- **Purpose:** Handling scanned images and providing bounding-box coordinate tracking for texts without digital layers.
- **Accuracy/Metrics:** Yields ~85-95% character accuracy depending on the DPI and quality of the scanned document. Confidence threshold is configured to `0.2` to capture as much technical text as possible during the initial pass.

### 2. Named Entity Recognition (NER)
- **Model:** `dslim/bert-base-NER`
- **Architecture:** BERT (Bidirectional Encoder Representations from Transformers) base model fine-tuned for NER.
- **Purpose:** Used to semantically extract equipment marks, tags, and measurements when simple regex fails.
- **Accuracy/Metrics:** Typically achieves 90%+ F1 score on standard entity recognition tasks. The application filters extractions with a confidence threshold $\ge$ `0.5`.

### 3. Document Layout Analysis (DLA)
- **Model:** `facebook/detr-resnet-50` (or `microsoft/layoutlmv3-base`)
- **Architecture:** DETR (DEtection TRansformer) with a ResNet-50 backbone, object detection pipeline.
- **Purpose:** To classify regions of the page into categories like 'Diagram', 'Schedule', 'Legend', and 'Text'. This identifies whether a page is purely architectural, mechanical, or just schedules.
- **Accuracy/Metrics:** High Average Precision (AP) for structural detection on standard document datasets (like PubLayNet). Configured with a `0.4` confidence threshold.

### 4. Table Structure Recognition
- **Model:** `microsoft/table-transformer-structure-recognition`
- **Architecture:** Table Transformer (DETR-based).
- **Purpose:** Precisely identifies rows, columns, and spanning cells in complex schedules or measurement tables.
- **Accuracy/Metrics:** Excellent performance (~90% mAP) on fully bordered tables. Configured with a `0.5` confidence threshold.

## System Features and Highlights
- **Conditional Highlighting:** Only modifies and highlights 'Mechanical Diagram' pages to save processing time and reduce false positives.
- **Cross-Referencing:** Upload an Excel file containing expected tags, and the system will locate and color-code matches directly on the diagram.
- **GPU Acceleration:** Intelligent device mapping dynamically routes operations to CUDA or MPS (Apple Silicon) when available for 10x-50x speedups.

## Running the Application
Ensure dependencies are installed:
```bash
pip install -r requirements.txt
```
Run the application via Streamlit:
```bash
streamlit run app.py
```
*(Optionally use `app2.py` or `advanced_extraction/app_advanced.py` for specific module testing).*
