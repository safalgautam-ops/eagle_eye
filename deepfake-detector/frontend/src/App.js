import React, { useState } from 'react';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setFileName(selectedFile.name);
      setError('');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!file) {
      setError('Please select a video file first.');
      return;
    }
    
    setLoading(true);
    setResult(null);
    setError('');
    
    const formData = new FormData();
    formData.append('video', file);
    
    try {
      const response = await fetch('http://localhost:4000/api/detect', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }
      
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(`Failed to analyze video: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Deepfake Detector</h1>
        <p>Upload a video to check if it's a deepfake</p>
      </header>
      
      <main>
        <form onSubmit={handleSubmit}>
          <div className="file-upload">
            <label htmlFor="video-upload">
              {fileName ? fileName : 'Choose a video file'}
            </label>
            <input
              type="file"
              id="video-upload"
              accept="video/*"
              onChange={handleFileChange}
            />
          </div>
          
          <button 
            type="submit" 
            disabled={loading || !file}
          >
            {loading ? 'Analyzing...' : 'Detect Deepfake'}
          </button>
        </form>
        
        {error && <div className="error">{error}</div>}
        
        {loading && (
          <div className="loading">
            <p>Analyzing video... This may take a few moments.</p>
          </div>
        )}
        
        {result && (
          <div className="result">
            <h2>Analysis Results</h2>
            <p className={result.prediction === 'REAL' ? 'real' : 'fake'}>
              This video is likely: <strong>{result.prediction}</strong>
            </p>
            <p>Confidence: {(result.confidence * 100).toFixed(2)}%</p>
            <p>Frames analyzed: {result.frames_analyzed}</p>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;