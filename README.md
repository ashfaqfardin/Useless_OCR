# EasyOCR Form App

![Project Banner](docs/image.png)

A modern, full-stack OCR playground and annotation tool. Upload images, extract text and bounding boxes using EasyOCR or Groq's vision models, edit and visualize results, and export structured data. Built with FastAPI, Next.js, and shadcn UI.

---

**Note: All OCR pipelines (EasyOCR, Groq, Hybrid, Prompt-based) are experimental and not fully functional. Use at your own risk and expect incomplete or inaccurate results.**

---

## Features

- Image Upload & Preview: Drag-and-drop or browse to upload images
- OCR Extraction: Use EasyOCR (local) or Groq's Llama 4 Maverick (cloud, vision)
- Bounding Box Visualization: See detected text regions with interactive boxes
- Edit & Annotate: Edit text, move/resize boxes, and correct results
- Zoom, Pan, Rotate: Modern viewer controls for large or complex images
- Language Selection: OCR in multiple languages, including Bangla
- Hybrid OCR: Combine Groq's text accuracy with EasyOCR's bounding boxes
- Prompt-based Extraction: Use custom prompts to extract structured data or bounding boxes
- Export: Download results as JSON
- Mobile Responsive: Works on desktop and mobile

---

## Screenshots

|     Upload & OCR      |          Bounding Boxes          |         Prompt Extraction         |
| :-------------------: | :------------------------------: | :-------------------------------: |
| ![UI](docs/image.png) | ![Exactration](docs/results.png) | ![Prompt](docs/boundingboxes.png) |

---

## Tech Stack

- Frontend: Next.js, React, TypeScript, shadcn UI
- Backend: FastAPI, Python, EasyOCR
- Cloud OCR: Groq API (Llama 4 Maverick Vision)
- Styling: Tailwind CSS

---

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+
- (Optional) Groq API key for cloud OCR

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/easyocr-form-app.git
cd easyocr-form-app
```

### 2. Backend Setup (FastAPI)

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

### 3. Frontend Setup (Next.js)

```bash
cd ../frontend
npm install
npm run dev
```

### 4. Open the App

Visit [http://localhost:3000](http://localhost:3000) in your browser.

---

## API Endpoints

### OCR

- `POST /ocr` — EasyOCR extraction (local)
- `POST /groq-ocr` — Groq vision model extraction (cloud)
- `POST /hybrid-ocr` — Hybrid: Groq text + EasyOCR boxes
- `POST /groq-prompt` — Custom prompt extraction (no OCR)
- `POST /groq-bounding-boxes` — Prompt-based bounding box extraction (JSON)
- `GET /groq-models` — List available Groq models

### Example: Prompt for Bounding Boxes

```json
{
  "bounding_boxes": [
    {
      "x1": 10,
      "y1": 20,
      "x2": 100,
      "y2": 60,
      "text": "Sample",
      "confidence": 0.95
    }
  ]
}
```

---

## Contribution

1. Fork the repo
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## License

[MIT License](LICENSE)

---

## Acknowledgements

- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [Groq](https://console.groq.com/)
- [shadcn UI](https://ui.shadcn.com/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Next.js](https://nextjs.org/)

---

## Contact

For questions, suggestions, or support, open an issue or contact [imashfaqfardin@gmail.com](mailto:imashfaqfardin@gmail.com).
