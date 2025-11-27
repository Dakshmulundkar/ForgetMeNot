const fs = require('fs');
const https = require('https');
const path = require('path');

// Create models directory if it doesn't exist
const modelsDir = path.join(__dirname, 'public', 'models');
if (!fs.existsSync(modelsDir)) {
  fs.mkdirSync(modelsDir, { recursive: true });
}

// Essential model files for face-api.js (core functionality)
const essentialModels = [
  // Face landmark detection
  {
    url: 'https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/face_landmark_68_model-weights_manifest.json',
    filename: 'face_landmark_68_model-weights_manifest.json'
  },
  {
    url: 'https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/face_landmark_68_model-shard1',
    filename: 'face_landmark_68_model-shard1'
  },
  
  // Face recognition (requires both shards)
  {
    url: 'https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/face_recognition_model-weights_manifest.json',
    filename: 'face_recognition_model-weights_manifest.json'
  },
  {
    url: 'https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/face_recognition_model-shard1',
    filename: 'face_recognition_model-shard1'
  },
  {
    url: 'https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/face_recognition_model-shard2',
    filename: 'face_recognition_model-shard2'
  },
  
  // SSD MobileNet v1 (used in the code)
  {
    url: 'https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/ssd_mobilenetv1_model-weights_manifest.json',
    filename: 'ssd_mobilenetv1_model-weights_manifest.json'
  },
  {
    url: 'https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/ssd_mobilenetv1_model-shard1',
    filename: 'ssd_mobilenetv1_model-shard1'
  },
  {
    url: 'https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/ssd_mobilenetv1_model-shard2',
    filename: 'ssd_mobilenetv1_model-shard2'
  },
  
  // Tiny face detector (lightweight option)
  {
    url: 'https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/tiny_face_detector_model-weights_manifest.json',
    filename: 'tiny_face_detector_model-weights_manifest.json'
  },
  {
    url: 'https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/tiny_face_detector_model-shard1',
    filename: 'tiny_face_detector_model-shard1'
  }
];

// Download function with error handling
function downloadFile(url, filepath) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(filepath);
    console.log(`Downloading ${url}...`);
    
    https.get(url, (response) => {
      if (response.statusCode === 404) {
        console.log(`File not found (404): ${url} - skipping...`);
        file.close();
        fs.unlink(filepath, () => {}); // Delete partial file
        resolve(); // Resolve successfully even if file is missing
        return;
      }
      
      if (response.statusCode !== 200) {
        file.close();
        fs.unlink(filepath, () => {}); // Delete partial file
        reject(new Error(`Failed to download ${url}: ${response.statusCode}`));
        return;
      }
      
      response.pipe(file);
      
      file.on('finish', () => {
        file.close();
        console.log(`Downloaded ${filepath}`);
        resolve();
      });
      
      file.on('error', (err) => {
        fs.unlink(filepath, () => {}); // Delete partial file
        reject(err);
      });
    }).on('error', (err) => {
      file.close();
      fs.unlink(filepath, () => {}); // Delete partial file
      reject(err);
    });
  });
}

// Download essential models
async function downloadModels() {
  console.log('Downloading essential face-api.js models...');
  
  try {
    for (const model of essentialModels) {
      const filepath = path.join(modelsDir, model.filename);
      try {
        await downloadFile(model.url, filepath);
      } catch (error) {
        console.error(`Error downloading ${model.filename}:`, error.message);
        // Continue with other models even if one fails
      }
    }
    console.log('Essential models download process completed!');
  } catch (error) {
    console.error('Error in download process:', error);
    process.exit(1);
  }
}

downloadModels();