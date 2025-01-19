# Flask 3D Model Generation API

This Flask application provides an API to generate 3D models from text prompts, convert them to specific formats, and serve them for download. It integrates with external APIs for model generation and supports additional functionality like file conversion and zipping.

---

## Features

- **Generate 3D Models**: Generate 3D models based on user-provided text prompts.
- **File Conversion**: Convert generated files from FBX to OBJ format using PyMeshLab.
- **File Management**: Zip generated files for easy download.
- **Download Endpoint**: Serve files through an API endpoint.

---

## Prerequisites

- Python 3.8+
- pip (Python package manager)

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Install Dependencies**:
   Install all required libraries using `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup Output Directory**:
   Ensure an `output` directory exists in the project root:
   ```bash
   mkdir output
   ```

---

## Usage

### Run the Application

Start the Flask server:
```bash
python app.py
```

The server will be available at `http://0.0.0.0:5000`.

### API Endpoints

#### **1. Generate Model**

**Endpoint:**
```http
POST /generate
```

**Request Body (JSON):**
```json
{
  "URID": "unique_model_id",
  "prompt": "A futuristic spaceship",
  "filename": "spaceship_model.zip"
}
```

**Response (JSON):**
```json
{
  "message": "Model generated successfully. Files saved and zipped at ./output/spaceship_model.zip",
  "zip_file": "./output/spaceship_model.zip"
}
```

#### **2. Download File**

**Endpoint:**
```http
GET /download/<filename>
```

**Example:**
Download the generated ZIP file:
```bash
curl -O http://localhost:5000/download/spaceship_model.zip
```

**Response:**
The requested file will be downloaded.

---

## File Structure

```
.
├── app.py             # Main Flask application
├── requirements.txt   # Python dependencies
├── output/            # Directory for generated files
├── README.md          # Documentation
```

---

## Dependencies

- Flask
- requests
- PyMeshLab

Install all dependencies using the provided `requirements.txt`.

---

## Notes

- Ensure your external model generation API is accessible and properly configured.
- Use absolute paths for directory management if running the app in different environments.
- Test endpoints using tools like Postman, cURL, or Unity.

---

## License

This project is licensed under the MIT License.

