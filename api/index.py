from flask import Flask, redirect, Response

app = Flask(__name__)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    # Simple redirect approach
    return redirect("https://proposal-assistant.onrender.com", code=302)

# For local development only
if __name__ == '__main__':
    app.run(debug=True) 