import { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [recognizedText, setRecognizedText] = useState("");
  const [selectedModel, setSelectedModel] = useState("combined");

  useEffect(() => {
    // Poll the backend every 1 second for the recognized text
    const interval = setInterval(() => {
      fetch("/recognized_text")
        .then(response => response.json())
        .then(data => {
          setRecognizedText(data.recognized_text);
        })
        .catch(error => console.error("Error fetching recognized text:", error));
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  // Function to clear the recognized text by calling the backend endpoint
  const handleClear = () => {
    fetch("/clear_text", {
      method: "POST",
    })
      .then(response => response.json())
      .then(data => {
        setRecognizedText(data.recognized_text);
      })
      .catch(error => console.error("Error clearing recognized text:", error));
  };

  const handleModelChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const newModel = event.target.value;
    setSelectedModel(newModel);

    fetch("/set_model_type", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ model_type: newModel }),
    })
      .then(response => response.json())
      .then(data => {
        if (data.status !== "success") {
          console.error("Failed to change model:", data.message);
        }
      })
      .catch(error => console.error("Error setting model type:", error));
  };

  return (
    <div>
      <header>
        <h1>ASL Recognition</h1>
      </header>
      <div className="video-container">
        <img src="/video_feed" alt="Video Feed" />
      </div>
      <div className="recognized-text">
        <h2>Detected Text:</h2>
        <p>{recognizedText}</p>
        <button className="clear-btn" onClick={handleClear}>
          Clear
        </button>
        <div className="model-selector">
          <label htmlFor="model-select">Select Model:</label>
          <select id="model-select" value={selectedModel} onChange={handleModelChange}>
            <option value="mlp">MLP</option>
            <option value="gcn">GCN</option>
            <option value="cnn">CNN</option>
            <option value="combined">Combined</option>
          </select>
        </div>
      </div>
    </div>
  );
}

export default App;
