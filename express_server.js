/*
 * Node.js + Express bridge for ReConPatch inference
 *
 * This small server exposes a REST API that accepts image uploads at
 * `/api/scan`, forwards them to the FastAPI inference microservice,
 * and returns the resulting JSON response to the client.  It uses
 * Multer for handling multipart form data and axios for making
 * HTTP requests to the Python backend.  Uploaded images are stored
 * temporarily on disk in the `uploads` directory and removed after
 * the request completes to avoid cluttering the filesystem.
 *
 * To run this server:
 *   npm install
 *   node express_server.js
 *
 * By default the server listens on port 3001.  Adjust the
 * `PORT` constant if needed.  The FastAPI service should be
 * listening on `FASTAPI_URL` (http://localhost:8000/infer by default).
 */

const express = require('express');
const multer = require('multer');
const axios = require('axios');
const fs = require('fs');
const FormData = require('form-data');

// Configuration
const PORT = process.env.PORT || 3001;
const FASTAPI_URL = process.env.FASTAPI_URL || 'http://localhost:8000/infer';

// Prepare an uploads directory for temporary files
const UPLOAD_DIR = 'uploads';
if (!fs.existsSync(UPLOAD_DIR)) {
  fs.mkdirSync(UPLOAD_DIR);
}

const app = express();
const upload = multer({ dest: UPLOAD_DIR });

/*
 * POST /api/scan
 *
 * Accepts a single image file with the form field name `image`.  The
 * file is forwarded to the FastAPI inference service as
 * multipart/form-data using axios.  The response from FastAPI is
 * returned directly to the client.  Any errors during forwarding or
 * inference are propagated as HTTP 500 responses with an error
 * message.
 */
app.post('/api/scan', upload.any(), async (req, res) => {
  // Accept either 'image' or 'file' as the form field name
  const files = req.files || [];
  const file = files.find(f => f.fieldname === 'image' || f.fieldname === 'file') || files[0];
  if (!file) {
    return res.status(400).json({ error: "No image file provided. Use form field 'image' or 'file'." });
  }
  try {
    // Prepare form data for FastAPI request
    const form = new FormData();
    form.append('file', fs.createReadStream(file.path), {
      filename: file.originalname,
    });

    // Forward the file to FastAPI
    const response = await axios.post(FASTAPI_URL, form, {
      headers: form.getHeaders(),
      // Increase timeout in case inference is slow
      timeout: 30000,
    });

    // Clean up temporary upload
    fs.unlink(file.path, () => {});

    return res.json(response.data);
  } catch (error) {
    // Clean up the file on error
    fs.unlink(file.path, () => {});
    console.error('Error forwarding image to inference service:', error.message);
    return res.status(500).json({ error: 'Inference service call failed.' });
  }
});

// Health check endpoint
app.get('/api/health', (_req, res) => {
  return res.json({ status: 'ok' });
});

app.listen(PORT, () => {
  console.log(`Express server listening on port ${PORT}`);
});


