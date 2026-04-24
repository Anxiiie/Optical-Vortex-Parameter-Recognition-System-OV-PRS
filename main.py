#!/usr/bin/env python3
"""
Main application file for Laguerre-Gaussian optical vortex recognition

Runs Flask web server to work with pre-trained PyTorch neural networks.
The application provides a web interface for recognizing optical vortex parameters
(radial index n, azimuthal index m, topological charge TC)
from intensity images from camera or hard disk.
"""

import sys
import os
from app import app

def main():
    """
    Main application startup function.

    Starts Flask server on localhost:5000 with debug mode.
    The server handles requests from the web interface for:
    - Loading pre-trained PyTorch models
    - Real-time image capture from web camera
    - Loading images from hard disk
    - Recognizing optical vortex parameters
    - Exporting results to CSV format
    """
    print("Starting web server for optical vortex recognition...")
    print("Open your browser and navigate to: http://127.0.0.1:5000")

    # Start Flask application
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=True)

if __name__ == "__main__":
    main()
