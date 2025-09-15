/*
 * Express bridge for ReConPatch inference
 */

const express = require('express');
const cors = require('cors');
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
// Allow browser clients (e.g., Next.js dev server on :3000)
app.use(cors());
const upload = multer({ dest: UPLOAD_DIR });

// POST /api/scan â€” forward image to FastAPI
app.post('/api/scan', upload.any(), async (req, res) => {
  const files = req.files || [];
  const file = files.find(f => f.fieldname === 'image' || f.fieldname === 'file') || files[0];
  if (!file) {
    return res.status(400).json({ error: "No image file provided. Use form field 'image' or 'file'." });
  }
  try {
    const form = new FormData();
    form.append('file', fs.createReadStream(file.path), { filename: file.originalname });

    const response = await axios.post(FASTAPI_URL, form, {
      headers: form.getHeaders(),
      timeout: 30000,
    });

    fs.unlink(file.path, () => {});
    return res.json(response.data);
  } catch (error) {
    fs.unlink(file.path, () => {});
    const status = error.response?.status || 500;
    const body = error.response?.data || { error: 'Inference service call failed.' };
    return res.status(status).json(body);
  }
});

// Health check endpoint
app.get('/api/health', (_req, res) => {
  return res.json({ status: 'ok' });
});

app.listen(PORT, () => {
  console.log(`Express server listening on port ${PORT}`);
});


