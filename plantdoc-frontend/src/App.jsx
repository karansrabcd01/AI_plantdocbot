import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [imageResponse, setImageResponse] = useState(null);
  const [textResponse, setTextResponse] = useState(null);
  const [loading, setLoading] = useState(false);

  // Chatbot states
  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const chatContainerRef = useRef(null);

  // Auto-scroll chat to bottom
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [chatMessages]);

  const handleImageSubmit = async (e) => {
    e.preventDefault();
    const file = e.target.elements.image.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);

    setLoading(true);
    try {
      const res = await axios.post('http://localhost:8000/image-prediction', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setImageResponse(res.data);
    } catch (err) {
      setImageResponse({ error: err.message });
    }
    setLoading(false);
  };

  const handleTextSubmit = async (e) => {
    e.preventDefault();
    const text = e.target.elements.input.value.trim();
    if (!text) return;

    setLoading(true);
    try {
      const res = await axios.post('http://localhost:8000/text-prediction', { input: text });
      setTextResponse(res.data);
    } catch (err) {
      setTextResponse({ error: err.message });
    }
    setLoading(false);
  };

  const handleChatSubmit = async (e) => {
    e.preventDefault();
    if (!chatInput.trim() || chatLoading) return;

    const userMessage = chatInput.trim();
    setChatInput('');

    // Add user message to chat
    const newUserMessage = { role: 'user', content: userMessage };
    setChatMessages(prev => [...prev, newUserMessage]);
    setChatLoading(true);

    try {
      // Prepare conversation history
      const history = chatMessages.map(msg => ({
        role: msg.role,
        content: msg.content
      }));

      const res = await axios.post('http://localhost:8000/chatbot', {
        message: userMessage,
        conversation_history: history
      });

      // Add bot response to chat
      const botMessage = {
        role: 'assistant',
        content: res.data.response || 'Sorry, I could not process that.',
        status: res.data.status
      };
      setChatMessages(prev => [...prev, botMessage]);
    } catch (err) {
      const errorMessage = {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        status: 'error'
      };
      setChatMessages(prev => [...prev, errorMessage]);
    }
    setChatLoading(false);
  };

  const handleChatKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleChatSubmit(e);
    }
  };

  return (
    <div className="center-bg">
      <div className="app-card">
        {/* Header */}
        <div className="plantdoc-header">
          <span role="img" aria-label="plant" className="logo-emoji">🌿</span>
          <div className="header-content">
            <h1 className="big-heading">PlantDocBot</h1>
            <p className="subtitle">AI-Powered Plant Disease Detection & Care Assistant</p>
          </div>
        </div>

        {/* Features Grid */}
        <div className="features-grid">
          {/* Image Disease Detection */}
          <section>
            <h2>📸 Image Analysis</h2>
            <form onSubmit={handleImageSubmit}>
              <div className="form-group">
                <label htmlFor="image-upload">Upload Plant Image</label>
                <input
                  type="file"
                  id="image-upload"
                  name="image"
                  accept="image/*"
                  required
                />
              </div>
              <button type="submit" disabled={loading}>
                {loading ? 'Analyzing...' : 'Analyze Image'}
              </button>
            </form>
            {imageResponse && (
              <div className={imageResponse.error ? "error-box" : "result-box"}>
                {imageResponse.error ? (
                  <div>{imageResponse.error}</div>
                ) : (
                  <>
                    <strong>🔍 Analysis Results</strong>
                    <ul>
                      <li><strong>Disease:</strong> {imageResponse.label}</li>
                      <li><strong>Confidence:</strong> {(imageResponse.confidence * 100).toFixed(1)}%</li>
                      <li><strong>Recommendation:</strong> {imageResponse.recommendation}</li>
                    </ul>
                  </>
                )}
              </div>
            )}
          </section>

          {/* Text Disease Diagnosis */}
          <section>
            <h2>💬 Text Diagnosis</h2>
            <form onSubmit={handleTextSubmit}>
              <div className="form-group">
                <label htmlFor="text-input">Describe Symptoms</label>
                <textarea
                  id="text-input"
                  name="input"
                  placeholder="e.g., Yellow spots on leaves, wilting stems..."
                  required
                />
              </div>
              <button type="submit" disabled={loading}>
                {loading ? 'Analyzing...' : 'Diagnose'}
              </button>
            </form>
            {textResponse && (
              <div className={textResponse.error ? "error-box" : "result-box"}>
                {textResponse.error ? (
                  <div>{textResponse.error}</div>
                ) : (
                  <>
                    <strong>🔍 Diagnosis Results</strong>
                    <ul>
                      <li><strong>Disease:</strong> {textResponse.label}</li>
                      <li><strong>Confidence:</strong> {(textResponse.confidence * 100).toFixed(1)}%</li>
                      <li><strong>Recommendation:</strong> {textResponse.recommendation}</li>
                    </ul>
                  </>
                )}
              </div>
            )}
          </section>
        </div>

        {/* Chatbot Section */}
        <section className="chatbot-section">
          <h2>🤖 AI Plant Care Assistant</h2>

          <div className="chat-container" ref={chatContainerRef}>
            {chatMessages.length === 0 ? (
              <div className="empty-chat">
                <div className="empty-chat-icon">🌱</div>
                <div className="empty-chat-text">
                  Ask me anything about plant care, diseases, or gardening tips!
                </div>
              </div>
            ) : (
              chatMessages.map((msg, idx) => (
                <div key={idx} className={`chat-message ${msg.role === 'user' ? 'user' : 'bot'}`}>
                  <div className={`message-bubble ${msg.role === 'user' ? 'user' : 'bot'}`}>
                    {msg.content}
                  </div>
                </div>
              ))
            )}

            {chatLoading && (
              <div className="chat-message bot">
                <div className="typing-indicator">
                  <div className="typing-dot"></div>
                  <div className="typing-dot"></div>
                  <div className="typing-dot"></div>
                </div>
              </div>
            )}
          </div>

          <form onSubmit={handleChatSubmit}>
            <div className="chat-input-container">
              <div className="chat-input-wrapper">
                <textarea
                  className="chat-input"
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyPress={handleChatKeyPress}
                  placeholder="Type your question here..."
                  rows="1"
                  disabled={chatLoading}
                />
              </div>
              <button
                type="submit"
                className="chat-send-btn"
                disabled={chatLoading || !chatInput.trim()}
              >
                {chatLoading ? 'Sending...' : 'Send 🚀'}
              </button>
            </div>
          </form>
        </section>

        {loading && <div className="loading-text">⏳ Processing...</div>}
      </div>
    </div>
  );
}

export default App;
