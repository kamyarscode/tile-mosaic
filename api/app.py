import json
import time

from flask import Flask, request

app = Flask(__name__)

# Test endpoint here.

@app.route('/api/test')
def test():
    
    return "This is test endpoint."

@app.route('/api/voronoi')
def render_voronoi():
    pass

def main():
    app.run(port=5000, debug=False)
    print ("Server is running now.")

if __name__ == "__main__":

    main()