const express = require('express');
const cors = require('cors');
const multer = require('multer');
const axios = require('axios');
const path = require('path');
const fs = require('fs');

// Initialize Express app
const app = express();
const port = 4000;

// Middlewares
app.use(cors());
app.use(express.json());

// Create uploads directory if it doesn't exist
const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir);
}

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, 'uploads/');
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + path.extname(file.originalname));
  }
});

const upload = multer({ 
  storage: storage,
  limits: { fileSize: 100 * 1024 * 1024 }, // Limit to 100MB
});

// Routes
app.get('/', (req, res) => {
  res.send('Deepfake Detector API is running');
});

// Process video for deepfake detection
app.post('/api/detect', upload.single('video'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No video file uploaded' });
    }

    const filePath = req.file.path;
    
    // Create form data to send to Python backend
    const formData = new FormData();
    formData.append('video', fs.createReadStream(filePath));
    
    // Call the Python ML service
    const response = await axios.post('http://localhost:5000/detect', formData, {
      headers: {
        ...formData.getHeaders(),
      },
    });
    
    // Return the detection results
    return res.json(response.data);
  } catch (error) {
    console.error('Error processing video:', error);
    return res.status(500).json({ 
      error: 'Failed to process video',
      details: error.message 
    });
  }
});

// Start the server
app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});