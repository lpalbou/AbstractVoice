#!/usr/bin/env python3
"""
Web API example using AbstractVoice with Flask.

This example shows how to create a simple web API that exposes
AbstractVoice functionality to web applications.
"""

import argparse
import json
import os
import tempfile
import uuid
from flask import Flask, request, jsonify, send_file, render_template_string

# Import VoiceManager only when needed
# from abstractvoice import VoiceManager


# Initialize Flask app
app = Flask(__name__)

# Global voice manager
voice_manager = None

# Store active sessions (in a real app, use a database)
active_sessions = {}


# Simple HTML template for the home page
HOME_PAGE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AbstractVoice Web API</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }
        h1, h2 {
            color: #333;
        }
        pre {
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
        }
        code {
            background-color: #f5f5f5;
            padding: 2px 4px;
            border-radius: 4px;
        }
        .endpoint {
            margin-bottom: 20px;
            border-bottom: 1px solid #eee;
            padding-bottom: 20px;
        }
    </style>
</head>
<body>
    <h1>AbstractVoice Web API</h1>
    <p>Welcome to the AbstractVoice Web API. Below are the available endpoints:</p>
    
    <div class="endpoint">
        <h2>GET /api/status</h2>
        <p>Get the status of the voice services.</p>
        <pre>curl http://{{ host }}:{{ port }}/api/status</pre>
    </div>
    
    <div class="endpoint">
        <h2>POST /api/tts</h2>
        <p>Convert text to speech and return audio file.</p>
        <p><strong>Request Body:</strong></p>
        <pre>{
    "text": "Text to speak",
    "speed": 1.0  // Optional
}</pre>
        <p><strong>Example:</strong></p>
        <pre>curl -X POST http://{{ host }}:{{ port }}/api/tts \
    -H "Content-Type: application/json" \
    -d '{"text":"Hello, this is a test", "speed":1.0}' \
    --output speech.wav</pre>
    </div>
    
    <div class="endpoint">
        <h2>POST /api/stt/transcribe</h2>
        <p>Transcribe audio from file.</p>
        <p><strong>Example:</strong></p>
        <pre>curl -X POST http://{{ host }}:{{ port }}/api/stt/transcribe \
    -F "audio_file=@/path/to/audio.wav"</pre>
    </div>
    
    <div class="endpoint">
        <h2>POST /api/stt/start</h2>
        <p>Start a listening session.</p>
        <pre>curl -X POST http://{{ host }}:{{ port }}/api/stt/start</pre>
    </div>
    
    <div class="endpoint">
        <h2>POST /api/stt/stop</h2>
        <p>Stop a listening session.</p>
        <p><strong>Request Body:</strong></p>
        <pre>{
    "session_id": "UUID of the session"
}</pre>
    </div>
</body>
</html>
"""


@app.route('/')
def home():
    """Serve the home page with API documentation."""
    host = request.host.split(':')[0]
    port = request.host.split(':')[1] if ':' in request.host else "5000"
    return render_template_string(HOME_PAGE_TEMPLATE, host=host, port=port)


@app.route('/api/test', methods=['GET'])
def test_api():
    """Simple test endpoint to verify the API is working."""
    return jsonify({
        "status": "ok",
        "message": "AbstractVoice Web API is running",
        "is_voice_manager_initialized": voice_manager is not None
    })


@app.route('/api/simpletest', methods=['GET'])
def simple_test():
    """A very simple test that doesn't require any initialization."""
    return jsonify({
        "status": "ok",
        "message": "Basic Flask API is working!",
        "timestamp": str(uuid.uuid4())
    })


# Simplified function that doesn't actually load the VoiceManager
def lazy_initialize_voice_manager(debug_mode=False):
    """Initialize the voice manager only when needed."""
    print("This is a placeholder for VoiceManager initialization")
    print("For a full implementation, uncomment the VoiceManager import")
    return None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="AbstractVoice Web API Example")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--host", default="127.0.0.1", help="Host to listen on")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    parser.add_argument("--simulate", action="store_true", 
                       help="Simulate only, don't load models")
    return parser.parse_args()


def main():
    """Entry point for the application."""
    global voice_manager
    
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Configure logging
        import logging
        log_level = logging.DEBUG if args.debug else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Configure Flask for development
        if args.debug:
            app.debug = True
            app.logger.setLevel(logging.DEBUG)
        else:
            app.logger.setLevel(logging.INFO)
        
        # Print startup message
        print(f"Starting AbstractVoice Web API on {args.host}:{args.port}")
        
        if not args.simulate:
            print("Initializing VoiceManager (this may take a moment)...")
            # Initialize voice manager - for real implementation, uncomment this
            # from abstractvoice import VoiceManager
            # voice_manager = VoiceManager(debug_mode=args.debug)
        else:
            print("Running in simulation mode (no models loaded)")
        
        # Run Flask app
        print(f"Server is ready at http://{args.host}:{args.port}")
        print("Try these test endpoints:")
        print(f"  http://{args.host}:{args.port}/")
        print(f"  http://{args.host}:{args.port}/api/simpletest")
        print("Press CTRL+C to quit")
        
        app.run(host=args.host, port=args.port)
        
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Application error: {e}")


if __name__ == "__main__":
    main() 